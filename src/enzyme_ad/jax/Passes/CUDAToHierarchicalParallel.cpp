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

  struct ParallelLoopTilingPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    int targetMaxThreads;
    int targetBitWidth;
    int targetUnrollFactor;
    
    ParallelLoopTilingPattern(MLIRContext *context, int bitWidth, int unrollFactor, int maxThreads)
      : OpRewritePattern<scf::ParallelOp>(context),
	targetBitWidth(bitWidth), targetUnrollFactor(unrollFactor),
        targetMaxThreads(maxThreads) {}

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      if (op->hasAttr("tiled") || op->hasAttr("vectorized")) return failure();

      Location loc = op.getLoc();
      Value ub = op.getUpperBound()[0];
      Value lb = op.getLowerBound()[0];
      Value totalIters = rewriter.create<arith::SubIOp>(loc, ub, lb);
      Value numThreads = rewriter.create<arith::ConstantIndexOp>(loc, targetMaxThreads);

      Value one = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      Value threadChunk = rewriter.create<arith::DivUIOp>(loc, rewriter.create<arith::AddIOp>(loc, totalIters,  
											      rewriter.create<arith::SubIOp>(loc, numThreads, one)),numThreads);
      
      Type elementType = nullptr;
      for (auto &innerOp : *op.getBody()) {
        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) 
          elementType = cast<MemRefType>(load.getMemref().getType()).getElementType();
        if (elementType) break;
      }
      if (!elementType) elementType = rewriter.getF32Type();
    
      int64_t lanes = targetBitWidth / elementType.getIntOrFloatBitWidth();
      if (lanes <= 1) return failure();
      
      int64_t vectorFloorElements = lanes * targetUnrollFactor;
      Value vectorFloor = rewriter.create<arith::ConstantIndexOp>(loc, vectorFloorElements);
      Value tileSize = rewriter.create<arith::MaxSIOp>(loc, threadChunk, vectorFloor);
            
      auto tiledLoop = rewriter.create<scf::ParallelOp>(loc, op.getLowerBound(), op.getUpperBound(), ValueRange{tileSize}, op.getInitVals());
      rewriter.eraseOp(tiledLoop.getBody()->getTerminator());
      rewriter.setInsertionPointToStart(tiledLoop.getBody());
        
      Value iv = tiledLoop.getInductionVars()[0];
      Value upper = rewriter.create<arith::MinUIOp>(loc, rewriter.create<arith::AddIOp>(loc, iv, tileSize), op.getUpperBound()[0]); 
        
      auto innerLoop = rewriter.create<scf::ParallelOp>(loc, ValueRange{iv}, ValueRange{upper}, op.getStep(), tiledLoop.getRegionIterArgs());
      rewriter.eraseOp(innerLoop.getBody()->getTerminator());

      innerLoop->setAttr("tiled", rewriter.getUnitAttr());
      tiledLoop->setAttr("tiled", rewriter.getUnitAttr());

      rewriter.mergeBlocks(op.getBody(), innerLoop.getBody(), innerLoop.getInductionVars());
      rewriter.setInsertionPointToEnd(tiledLoop.getBody());
      rewriter.create<scf::ReduceOp>(loc, innerLoop.getResults());

      rewriter.replaceOp(op, tiledLoop.getResults());
      return success();
    }
  };

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

  struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    int targetBitWidth;
    int targetUnrollFactor;

    CUDAToVectorPattern(MLIRContext *context, int bitWidth, int unrollFactor)
      : OpRewritePattern<scf::ParallelOp>(context), 
        targetBitWidth(bitWidth), targetUnrollFactor(unrollFactor) {}

    static std::optional<vector::CombiningKind> getReductionKindFromOp(Operation *op) {
      if (isa<arith::AddFOp, arith::AddIOp>(op)) return vector::CombiningKind::ADD;
      if (isa<arith::MulFOp, arith::MulIOp>(op)) return vector::CombiningKind::MUL;
      if (isa<arith::MaximumFOp, arith::MaxSIOp, arith::MaxUIOp>(op)) return vector::CombiningKind::MAXSI;
      if (isa<arith::MinimumFOp, arith::MinSIOp, arith::MinUIOp>(op)) return vector::CombiningKind::MINSI;
      return std::nullopt;
    }

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      if (op->hasAttr("vectorized")) return failure();
      if (op.getLowerBound().size() != 1) return failure();

      bool containsParallel = false;
      op.getRegion().walk([&](scf::ParallelOp nested) {
        if (nested != op) containsParallel = true;
      });
      if (containsParallel) return failure();

      Location loc = op.getLoc();
      Type elementType = nullptr;
      for (auto &innerOp : *op.getBody()) {
        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) 
          elementType = cast<MemRefType>(load.getMemref().getType()).getElementType();
        if (elementType) break;
      }
      if (!elementType) elementType = rewriter.getF32Type();
    
      int64_t vWidth = targetBitWidth / elementType.getIntOrFloatBitWidth();
      if (vWidth <= 1) return failure();

      VectorType vType = VectorType::get({vWidth}, elementType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
      int64_t totalStep = vWidth * targetUnrollFactor;
      Value stepConst = rewriter.create<arith::ConstantIndexOp>(loc, totalStep);

      vector::CombiningKind redKind = vector::CombiningKind::ADD;
      if (op.getNumReductions() > 0) {
	auto reduceOps = op.getBody()->getOps<scf::ReduceOp>();
	if (!reduceOps.empty()) {
	  scf::ReduceOp firstReduce = *reduceOps.begin();
	  Block &redBlock = firstReduce.getRegion().front();
	  if (!redBlock.empty() && !redBlock.front().hasTrait<OpTrait::IsTerminator>()) {
            if (auto kind = getReductionKindFromOp(&redBlock.front()))
	      redKind = *kind;
	  }
	}
      }
      
      SmallVector<Value> vectorInits;
      for (Type t : op.getResultTypes()) {
        Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(t));
        vectorInits.push_back(rewriter.create<vector::BroadcastOp>(loc, VectorType::get({vWidth}, t), zero));
      }

      auto newLoop = rewriter.create<scf::ParallelOp>(loc, op.getLowerBound(), op.getUpperBound(), ValueRange{stepConst}, vectorInits);
      newLoop->setAttr("vectorized", rewriter.getUnitAttr());

      Block *loopBody = newLoop.getBody();
      rewriter.eraseOp(loopBody->getTerminator());
      rewriter.setInsertionPointToStart(loopBody);

      Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ub = newLoop.getUpperBound()[0];
      Value cstZero = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(elementType));
      SmallVector<Value> partialSums(newLoop.getRegionIterArgs().begin(), newLoop.getRegionIterArgs().end());
      auto identityMap = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));

      for (int u = 0; u < targetUnrollFactor; ++u) {
        IRMapping mapping;
        Value offset = rewriter.create<arith::ConstantIndexOp>(loc, u * vWidth);
        Value iv = rewriter.create<arith::AddIOp>(loc, newLoop.getInductionVars()[0], offset);
        mapping.map(op.getInductionVars()[0], iv);

        Value diff = rewriter.create<arith::SubIOp>(loc, ub, iv);
        Value clamped = rewriter.create<arith::MaxSIOp>(loc, diff, zeroIdx);
        Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, clamped);

        for (Operation &innerOp : op.getBody()->getOperations()) {
          if (innerOp.hasTrait<OpTrait::IsTerminator>()) {
            if (auto reduceOp = dyn_cast<scf::ReduceOp>(innerOp)) {
              for (auto [i, operand] : llvm::enumerate(reduceOp.getOperands())) {
                Value val = mapping.lookupOrDefault(operand);
                if (!llvm::isa<VectorType>(val.getType()))
                  val = rewriter.create<vector::BroadcastOp>(loc, vType, val);
                
                // --- FIX: Accumulate with correct op ---
                if (llvm::isa<FloatType>(elementType)) {
		  if (redKind == vector::CombiningKind::MUL)
		    partialSums[i] = rewriter.create<arith::MulFOp>(loc, partialSums[i], val);
		  else
		    partialSums[i] = rewriter.create<arith::AddFOp>(loc, partialSums[i], val);
                } else {
		  if (redKind == vector::CombiningKind::MUL)
		    partialSums[i] = rewriter.create<arith::MulIOp>(loc, partialSums[i], val);
		  else
		    partialSums[i] = rewriter.create<arith::AddIOp>(loc, partialSums[i], val);
                }
              }
            }
            continue;
          }

          if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
            SmallVector<Value> idxs;
            for (auto i : load.getIndices()) idxs.push_back(mapping.lookupOrDefault(i));
            mapping.map(load.getResult(), rewriter.create<vector::TransferReadOp>(
										  loc, vType, mapping.lookupOrDefault(load.getMemref()), idxs, 
										  identityMap, cstZero, mask, rewriter.getBoolArrayAttr({false})));
          } 
          else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
            Value val = mapping.lookupOrDefault(store.getValueToStore());
            if (!llvm::isa<VectorType>(val.getType())) 
              val = rewriter.create<vector::BroadcastOp>(loc, vType, val);
            SmallVector<Value> idxs;
            for (auto i : store.getIndices()) idxs.push_back(mapping.lookupOrDefault(i));
            rewriter.create<vector::TransferWriteOp>(
						     loc, val, mapping.lookupOrDefault(store.getMemref()), idxs, 
						     identityMap, mask, rewriter.getBoolArrayAttr({false}));
          }
          else if (innerOp.getDialect()->getNamespace() == "arith" || innerOp.getDialect()->getNamespace() == "math") {
            SmallVector<Value> ops;
            for (auto o : innerOp.getOperands()) {
              Value v = (o == op.getInductionVars()[0]) ? iv : mapping.lookupOrDefault(o);
              if (!llvm::isa<VectorType>(v.getType())) v = rewriter.create<vector::BroadcastOp>(loc, vType, v);
              ops.push_back(v);
            }
            OperationState state(loc, innerOp.getName().getStringRef());
            state.addOperands(ops);
            for (Type t : innerOp.getResultTypes()) state.addTypes(VectorType::get({vWidth}, t));
            state.addAttributes(innerOp.getAttrs());
            Operation *vOp = rewriter.create(state);
            for (unsigned i = 0; i < innerOp.getNumResults(); ++i) mapping.map(innerOp.getResult(i), vOp->getResult(i));
          }
        }
      }

      rewriter.setInsertionPointToEnd(loopBody);
      rewriter.create<scf::ReduceOp>(loc, partialSums);

      rewriter.setInsertionPointAfter(newLoop);
      if (newLoop.getNumResults() > 0) {
        SmallVector<Value> scalarResults;
        for (Value vRes : newLoop.getResults()) {
          scalarResults.push_back(rewriter.create<vector::ReductionOp>(loc, redKind, vRes));
        }
        rewriter.replaceOp(op, scalarResults);
      } else {
        rewriter.eraseOp(op);
      }
      return success();
    }
  };
    
  struct CUDAToHierarchicalParallelPass : public enzyme::impl::CUDAToHierarchicalParallelBase<CUDAToHierarchicalParallelPass>  {
    using Base::Base;
  
    void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *ctx = &getContext();

      int finalWidth = (regBitWidth > 0) ? regBitWidth : 256; 
      int finalUnrollFactor = (unrollFactor > 0) ? unrollFactor : 4;

      if (auto attr = module->getAttrOfType<StringAttr>("llvm.target_features")) {
        llvm::StringRef features = attr.getValue();
        if (features.contains("+avx512f")) finalWidth = 512;
        else if (features.contains("+avx2") || features.contains("+avx")) finalWidth = 256;
      }

      // Move gpu.allocs to memref.allocas
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

      RewritePatternSet collapsePatterns(ctx);
      collapsePatterns.add<ParallelLoopCollapsePattern>(ctx);
      (void)applyPatternsGreedily(module, std::move(collapsePatterns));
      
      RewritePatternSet tilePatterns(ctx);
      tilePatterns.add<ParallelLoopTilingPattern>(ctx, finalWidth, finalUnrollFactor, maxThreads);
      (void)applyPatternsGreedily(module, std::move(tilePatterns));
      
      RewritePatternSet prePatterns(ctx);
      prePatterns.add<BarrierFissionPattern>(ctx);
      (void)applyPatternsGreedily(module, std::move(prePatterns));
      
      RewritePatternSet patterns(ctx);
      patterns.add<CUDAToVectorPattern>(ctx, finalWidth, finalUnrollFactor);
      (void)applyPatternsGreedily(module, std::move(patterns));
    }
  };
    
}
