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

      rewriter.mergeBlocks(op.getBody(), newLoop.getBody(), decodedIvs);
        
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
      : OpRewritePattern<scf::ParallelOp>(context) {
      this->targetBitWidth = bitWidth; 
    }

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      if (op.getInductionVars().size() != 1) return failure();
      if (op->hasAttr("vectorized")) return failure();

      Location loc = op.getLoc();
      Type elementType = nullptr;
      for (auto &innerOp : *op.getBody()) {
        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) 
	  elementType = cast<MemRefType>(load.getMemref().getType()).getElementType();
        else if (auto store = dyn_cast<memref::StoreOp>(innerOp))
	  elementType = cast<MemRefType>(store.getMemref().getType()).getElementType();
        if (elementType) break;
      }
      if (!elementType) elementType = rewriter.getF32Type();

      int64_t vWidth = targetBitWidth / elementType.getIntOrFloatBitWidth();
      if (vWidth <= 1) return failure();

      VectorType vType = VectorType::get({vWidth}, elementType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
    
      Value vWidthConst = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
      auto newLoop = rewriter.create<scf::ParallelOp>(
						      loc, op.getLowerBound(), op.getUpperBound(), ValueRange{vWidthConst}, op.getInitVals());
      newLoop->setAttr("vectorized", rewriter.getUnitAttr());

      Block *newBody = newLoop.getBody();
      rewriter.setInsertionPointToStart(newBody);

      Value scalarIv = newLoop.getInductionVars()[0];
      Value ub = newLoop.getUpperBound()[0];

      IRMapping mapping;
      mapping.map(op.getInductionVars()[0], scalarIv);

      Value baseIv = rewriter.create<arith::IndexCastOp>(loc, elementType, scalarIv);
      Value splatIv = rewriter.create<vector::BroadcastOp>(loc, vType, baseIv);
    
      SmallVector<Attribute> offsetAttrs;
      for (int64_t i = 0; i < vWidth; ++i) {
        if (auto fType = llvm::dyn_cast<FloatType>(elementType)) 
	  offsetAttrs.push_back(FloatAttr::get(fType, (double)i));
        else 
	  offsetAttrs.push_back(IntegerAttr::get(elementType, i));
      }
      Value offsets = rewriter.create<arith::ConstantOp>(loc, DenseElementsAttr::get(vType, offsetAttrs));
      Value vectorIv = llvm::isa<FloatType>(elementType) ? 
        rewriter.create<arith::AddFOp>(loc, splatIv, offsets).getResult() :
        rewriter.create<arith::AddIOp>(loc, splatIv, offsets).getResult();

      Value diff = rewriter.create<arith::SubIOp>(loc, ub, scalarIv);
      Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
      Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));

      for (Operation &innerOp : op.getBody()->getOperations()) {
        if (innerOp.hasTrait<OpTrait::IsTerminator>()) continue;

        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
	  SmallVector<Value> indices;
	  for (Value idx : load.getIndices()) 
	    indices.push_back(mapping.lookupOrDefault(idx));

	  auto vRead = rewriter.create<vector::TransferReadOp>(
							       loc, vType, mapping.lookupOrDefault(load.getMemref()), indices,
							       AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1)),
							       zeroPadding, mask, rewriter.getBoolArrayAttr({false}));
	  mapping.map(load.getResult(), vRead.getResult());
        } 
        else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
	  Value vecVal = mapping.lookupOrDefault(store.getValueToStore());
	  if (vecVal.getType() != vType)
	    vecVal = rewriter.create<vector::BroadcastOp>(loc, vType, vecVal);

	  SmallVector<Value> indices;
	  for (Value idx : store.getIndices()) 
	    indices.push_back(mapping.lookupOrDefault(idx));

	  rewriter.create<vector::TransferWriteOp>(
						   loc, vecVal, mapping.lookupOrDefault(store.getMemref()), indices,
						   AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1)),
						   mask, rewriter.getBoolArrayAttr({false}));
        }
        else if (innerOp.getDialect()->getNamespace() == "arith" || 
                 innerOp.getDialect()->getNamespace() == "math") {
            
	  SmallVector<Value> vOperands;
	  for (Value operand : innerOp.getOperands()) {
	    // If it's the IV, use the vectorIv we built
	    if (operand == op.getInductionVars()[0]) {
	      vOperands.push_back(vectorIv);
	      continue;
	    }
                
	    Value v = mapping.lookupOrDefault(operand);
	    if (!llvm::isa<VectorType>(v.getType()))
	      v = rewriter.create<vector::BroadcastOp>(loc, vType, v);
	    vOperands.push_back(v);
	  }

	  OperationState state(loc, innerOp.getName().getStringRef());
	  state.addOperands(vOperands);
	  for (Type t : innerOp.getResultTypes()) 
	    state.addTypes(VectorType::get({vWidth}, t));
	  state.addAttributes(innerOp.getAttrs());
            
	  Operation *vOp = rewriter.create(state);
	  for (unsigned i = 0; i < innerOp.getNumResults(); ++i)
	    mapping.map(innerOp.getResult(i), vOp->getResult(i));
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
