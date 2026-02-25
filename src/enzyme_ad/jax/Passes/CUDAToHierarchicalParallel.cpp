#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/EnzymeHLOPatterns.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"

#include <iostream>

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_CUDATOHIERARCHICALPARALLEL
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

  // =========================================================================
  // Pattern 1: Collapse 2D/3D Loops to 1D
  // Flattens N-Dimensional scf.parallel loops into a 1D loop.
  // This maps perfectly to linear CPU thread pools and linear SIMD vectors.
  // =========================================================================
struct ParallelLoopCollapsePattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
        unsigned numDims = op.getInductionVars().size();
        if (numDims <= 1) return failure();

        Location loc = op.getLoc();
        Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
        Value zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);

        Value totalSize = one;
        SmallVector<Value> sizes;

        for (unsigned i = 0; i < numDims; ++i) {
            Value diff = rewriter.create<arith::SubIOp>(loc, op.getUpperBound()[i], op.getLowerBound()[i]);
            Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, op.getStep()[i], one);
            Value size = rewriter.create<arith::DivUIOp>(loc, rewriter.create<arith::AddIOp>(loc, diff, stepMinusOne), op.getStep()[i]);
            sizes.push_back(size);
            totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, size);
        }

        auto newLoop = rewriter.create<scf::ParallelOp>(loc, zero, totalSize, one, op.getInitVals());
        rewriter.setInsertionPointToStart(newLoop.getBody());

        Value currentVal = newLoop.getInductionVars()[0];
        SmallVector<Value> decodedIvs(numDims);
        for (int i = numDims - 1; i >= 0; --i) {
            Value rem = rewriter.create<arith::RemUIOp>(loc, currentVal, sizes[i]);
            decodedIvs[i] = rewriter.create<arith::AddIOp>(loc, op.getLowerBound()[i], 
                             rewriter.create<arith::MulIOp>(loc, rem, op.getStep()[i]));
            if (i > 0) currentVal = rewriter.create<arith::DivUIOp>(loc, currentVal, sizes[i]);
        }

        // Use rewriter.mergeBlocks instead of manual splice
        rewriter.mergeBlocks(op.getBody(), newLoop.getBody(), decodedIvs);
        
        // Remove the old terminator from the merged block (it was the old scf.reduce/yield)
        rewriter.eraseOp(newLoop.getBody()->getTerminator()); 

        rewriter.replaceOp(op, newLoop.getResults());
        return success();
    }
};


  // =========================================================================
  // Pattern 2: Vectorisation & Strip-Mining
  // =========================================================================
  struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  int targetBitWidth;

  CUDAToVectorPattern(MLIRContext *context, int bitWidth)
      : OpRewritePattern<scf::ParallelOp>(context), targetBitWidth(bitWidth) {}

  LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
    bool hasNestedParallel = false;
    op.walk([&](scf::ParallelOp nested) { if (nested != op) hasNestedParallel = true; });
    if (hasNestedParallel) return failure();

    if (op.getInductionVars().size() != 1) return failure();
    // Guard: Prevent re-vectorising (Step > 1)
    if (auto constStep = op.getStep()[0].getDefiningOp<arith::ConstantIndexOp>()) {
      if (constStep.value() > 1) return failure();
    }

    auto loc = op.getLoc();
Type elementType = nullptr;

for (auto &innerOp : *op.getBody()) {
    if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
        if (auto mType = dyn_cast<MemRefType>(load.getMemref().getType()))
            elementType = mType.getElementType();
    } else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
        if (auto mType = dyn_cast<MemRefType>(store.getMemref().getType()))
            elementType = mType.getElementType();
    }
    if (elementType) break;
}

// Fallback if no loads/stores found
if (!elementType) elementType = rewriter.getF32Type();

unsigned bitWidth = elementType.getIntOrFloatBitWidth();
if (bitWidth == 0 || targetBitWidth % bitWidth != 0) {
    return rewriter.notifyMatchFailure(op, "Invalid element bitwidth");
}

int64_t vWidth = targetBitWidth / bitWidth;
if (vWidth <= 1) return failure();

    VectorType vType = VectorType::get({(int64_t)vWidth}, elementType);
    VectorType maskType = VectorType::get({(int64_t)vWidth}, rewriter.getI1Type());

    // Create the new vectorised parallel loop
    Value newStep = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vWidth));
      rewriter.modifyOpInPlace(op, [&]() {
      op.getStepMutable().assign(newStep);
    });

    auto newLoop = rewriter.create<scf::ParallelOp>(
        loc, op.getLowerBound(), op.getUpperBound(), ValueRange{newStep}, op.getInitVals());

    // Move insertion point to the new loop body
    rewriter.setInsertionPointToStart(newLoop.getBody());
    Value scalarIv = newLoop.getInductionVars()[0];
    Value ub = newLoop.getUpperBound()[0];

    // Thread ID Vectorisation
    Value baseThreadId;
    if (isa<FloatType>(elementType)) {
      Value intIv = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), scalarIv);
      baseThreadId = rewriter.create<arith::SIToFPOp>(loc, elementType, intIv);
    } else {
      baseThreadId = rewriter.create<arith::IndexCastOp>(loc, elementType, scalarIv);
    }
    Value splatIv = rewriter.create<vector::BroadcastOp>(loc, vType, baseThreadId);

    SmallVector<Attribute> offsetAttrs;
for (int64_t i = 0; i < vWidth; ++i) {
    if (auto floatType = dyn_cast<FloatType>(elementType)) {
        offsetAttrs.push_back(rewriter.getFloatAttr(floatType, (double)i));
    } else {
        offsetAttrs.push_back(rewriter.getIntegerAttr(elementType, i));
    }
}
    Value offsetsConst = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vType, offsetAttrs));
    Value vectorisedThreadId = isa<FloatType>(elementType) ? 
        rewriter.create<arith::AddFOp>(loc, splatIv, offsetsConst).getResult() :
        rewriter.create<arith::AddIOp>(loc, splatIv, offsetsConst).getResult();

    // Mapping from OLD values to new vectorised values
    DenseMap<Value, Value> vectorisedMapping;
    vectorisedMapping[op.getInductionVars()[0]] = vectorisedThreadId;

    Value diff = rewriter.create<arith::SubIOp>(loc, ub, scalarIv);
    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
    Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));

    // Helper to get vectorised version of an operand
   auto getVecOp = [&](Value scalarVal, Operation *contextOp) -> Value {
    if (vectorisedMapping.count(scalarVal)) {
        Value v = vectorisedMapping[scalarVal];
        if (v) return v;
    }

    // Context-aware insertion
    OpBuilder::InsertionGuard guard(rewriter);
    if (Operation *defOp = scalarVal.getDefiningOp()) {
        rewriter.setInsertionPointAfter(defOp);
    } else {
        rewriter.setInsertionPoint(contextOp);
    }

    Type type = scalarVal.getType();
    Value toSplat = scalarVal;
    if (type.isIndex()) {
        toSplat = rewriter.create<arith::IndexCastOp>(contextOp->getLoc(), rewriter.getI64Type(), scalarVal);
        type = rewriter.getI64Type();
    }

    VectorType splatVType = VectorType::get({vWidth}, type);
    Value splat = rewriter.create<vector::BroadcastOp>(contextOp->getLoc(), splatVType, toSplat);
    
    vectorisedMapping[scalarVal] = splat;
    return splat;
};


    // Build the new body by iterating the old one (WITHOUT replacing ops yet)
    for (Operation &innerOp : op.getBody()->getOperations()) {
      if (innerOp.hasTrait<OpTrait::IsTerminator>()) continue;

      if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
        auto vRead = rewriter.create<vector::TransferReadOp>(
            loc, vType, load.getMemref(), load.getIndices(),
            AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1)),
            zeroPadding, mask, rewriter.getBoolArrayAttr({false}));
        vectorisedMapping[load.getResult()] = vRead.getResult();
      } 
    else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
    rewriter.setInsertionPoint(store);

    Value vVal = getVecOp(store.getValueToStore(), &innerOp);
    if (!vVal) return failure();

    rewriter.create<vector::TransferWriteOp>(
        loc, vVal, store.getMemref(), store.getIndices(),
        AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1)),
        mask, rewriter.getBoolArrayAttr({false}));

    rewriter.eraseOp(store);
}
      else if (innerOp.getDialect()->getNamespace() == "arith") {
    // Only handle simple mapping of scalar-to-vector ops
    if (innerOp.getNumResults() == 1) {
        rewriter.setInsertionPoint(&innerOp);
        
        SmallVector<Value, 2> vOperands;
for (Value operand : innerOp.getOperands()) {
    // Pass &innerOp so the helper knows WHERE to create the broadcast
    Value vOp = getVecOp(operand, &innerOp); 
    if (!vOp) return failure();
    vOperands.push_back(vOp);
}
        Type scalarRetType = innerOp.getResult(0).getType();
        VectorType vRetType = VectorType::get({(int64_t)vWidth}, scalarRetType);

        OperationState state(innerOp.getLoc(), innerOp.getName().getStringRef());
        state.addOperands(vOperands);
        state.addTypes(vRetType);
        state.addAttributes(innerOp.getAttrs());

        Operation *vOp = rewriter.create(state);
        
        vectorisedMapping[innerOp.getResult(0)] = vOp->getResult(0);
        
        rewriter.replaceOp(&innerOp, vOp->getResults());
        continue;
    }
}
    }

    rewriter.replaceOp(op, newLoop.getResults());
    return success();
  }
};
  
  
  // =========================================================================
  // Pattern 3: Barrier Fission
  // =========================================================================
struct BarrierFissionPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp parallelOp, PatternRewriter &rewriter) const override {
        gpu::BarrierOp barrier;
        for (auto &op : *parallelOp.getBody()) {
            if (auto b = dyn_cast<gpu::BarrierOp>(&op)) { barrier = b; break; }
        }
        if (!barrier) return failure();

        rewriter.setInsertionPointAfter(parallelOp);
        auto secondLoop = cast<scf::ParallelOp>(rewriter.clone(*parallelOp.getOperation()));
        
        bool found = false;
        for (Operation &inner : llvm::make_early_inc_range(*parallelOp.getBody())) {
            if (&inner == barrier.getOperation()) { found = true; rewriter.eraseOp(&inner); continue; }
            if (found && !inner.hasTrait<OpTrait::IsTerminator>()) rewriter.eraseOp(&inner);
        }

        found = false;
        rewriter.setInsertionPointToStart(secondLoop.getBody());
        for (Operation &inner : llvm::make_early_inc_range(*secondLoop.getBody())) {
            if (auto b = dyn_cast<gpu::BarrierOp>(inner)) {
                found = true;
                rewriter.eraseOp(b);
                continue;
            }
            if (!found && !inner.hasTrait<OpTrait::IsTerminator>()) rewriter.eraseOp(&inner);
        }

        return success();
    }
};

  // =========================================================================
  // Pass Runner
  // =========================================================================
  struct CUDAToHierarchicalParallelPass : public enzyme::impl::CUDAToHierarchicalParallelBase<CUDAToHierarchicalParallelPass>  {
    using Base::Base;
  
    void runOnOperation() override {
      llvm::errs() << "RUNNING CUDA TO HIERARCHICAL PASS\n";

      ModuleOp module = getOperation();
      MLIRContext *ctx = &getContext();
 
      assert(regBitWidth > 0 && "expected positive bit width for vectorisation");
      assert((regBitWidth & (regBitWidth - 1)) == 0 && "expected the bit width for vectorisation to be a power of 2");

      int finalWidth = (regBitWidth > 0) ? regBitWidth : 256; 

      if (auto attr = module->getAttrOfType<StringAttr>("llvm.target_features")) {
        llvm::StringRef features = attr.getValue();
        if (features.contains("+avx512f")) finalWidth = 512;
        else if (features.contains("+avx2") || features.contains("+avx")) finalWidth = 256;
        else if (features.contains("+neon") || features.contains("+sse")) finalWidth = 128;
      }

      SmallVector<gpu::AllocOp> allocsToMove;
      module.walk([&](gpu::AllocOp alloc) {
        if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(alloc.getType().getMemorySpace())) {
          if (intAttr.getInt() == 3) allocsToMove.push_back(alloc);
        }
      });

      for (auto alloc : allocsToMove) {
        func::FuncOp parentFunc = alloc->getParentOfType<func::FuncOp>();
        if (!parentFunc) continue; 
        OpBuilder hoistedBuilder(&parentFunc.getBody().front(), parentFunc.getBody().front().begin());
        auto stackMem = hoistedBuilder.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
        alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
        alloc.erase();
      }

      // Flatten 2D/3D loops to 1D flat structures
      RewritePatternSet collapsePatterns(ctx);
      collapsePatterns.add<ParallelLoopCollapsePattern>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(collapsePatterns)))) signalPassFailure();

      // Resolve Barriers by Fissioning the loops
      RewritePatternSet prePatterns(ctx);
      prePatterns.add<BarrierFissionPattern>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(prePatterns)))) signalPassFailure();

      // Strip-Mine and Vectorise the Innermost loops
      RewritePatternSet patterns(ctx);
      patterns.add<CUDAToVectorPattern>(ctx, finalWidth);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) signalPassFailure();
    }
  };
    
}
