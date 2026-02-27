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
    int targetBitWidth;
    int targetUnrollFactor;

    CUDAToVectorPattern(MLIRContext *ctx, int bitWidth, int unrollFactor)
      : OpRewritePattern<scf::ParallelOp>(ctx), 
        targetBitWidth(bitWidth), 
        targetUnrollFactor(unrollFactor) {}

    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {
      //Guard against re-processing
      if (op->hasAttr("vectorized")) return failure();
      if (op.getLowerBound().size() != 1) return failure();

      Location loc = op.getLoc();

      // Check for nested parallel loops
      bool containsParallel = false;
      for (Operation &innerOp : op.getRegion().getOps()) {
	innerOp.walk([&](scf::ParallelOp nested) {
	  containsParallel = true;
	  return WalkResult::interrupt();
	});
	if (containsParallel) break;
      }
      if (containsParallel) return failure();

      // Dynamic vector width calculation
      Type elemType = rewriter.getF32Type();
      if (!op.getInitVals().empty()) {
	elemType = op.getInitVals()[0].getType();
      }
      unsigned elemBitWidth = elemType.getIntOrFloatBitWidth();
      int vWidth = targetBitWidth / (elemBitWidth > 0 ? elemBitWidth : 32);
      VectorType vType = VectorType::get({vWidth}, elemType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
      Value stepConst = rewriter.create<arith::ConstantIndexOp>(loc, vWidth * targetUnrollFactor);

      auto getReductionKindFromOp = [](Operation *mathOp) -> std::optional<vector::CombiningKind> {
	if (isa<arith::AddFOp, arith::AddIOp>(mathOp)) return vector::CombiningKind::ADD;
	if (isa<arith::MulFOp, arith::MulIOp>(mathOp)) return vector::CombiningKind::MUL;
	if (isa<arith::MaximumFOp, arith::MaxSIOp>(mathOp)) return vector::CombiningKind::MAXSI;
	if (isa<arith::MinimumFOp, arith::MinSIOp>(mathOp)) return vector::CombiningKind::MINSI;
	return std::nullopt;
      };

      // Reduction kind detection
      vector::CombiningKind redKind = vector::CombiningKind::ADD;
      if (op.getNumReductions() > 0) {
	auto reduceOps = op.getBody()->getOps<scf::ReduceOp>();
	if (!reduceOps.empty()) {
	  scf::ReduceOp firstReduce = *reduceOps.begin();
	  if (firstReduce && firstReduce.getNumRegions() > 0) {
	    Region &redRegion = firstReduce.getRegion(0);
	    if (!redRegion.empty()) {
	      auto it = redRegion.begin();
	      if (it != redRegion.end()) {
		Block &redBlock = *it;
		for (Operation &mathOp : llvm::make_early_inc_range(redBlock.getOperations())) {
		  if (!mathOp.hasTrait<OpTrait::IsTerminator>()) {
		    if (auto kind = getReductionKindFromOp(&mathOp)) {
		      redKind = *kind;
		      break;
		    }
		  }
		}
	      }
	    }
	  }
	}
      }

      SmallVector<Value> vectorInits;
      for (auto [i, t] : llvm::enumerate(op.getResultTypes())) {
	Attribute identityAttr;
	if (redKind == vector::CombiningKind::MUL) {
	  if (auto ft = llvm::dyn_cast<FloatType>(t))
	    identityAttr = rewriter.getFloatAttr(ft, 1.0);
	  else
	    identityAttr = rewriter.getIntegerAttr(t, 1);
	} else if (redKind == vector::CombiningKind::MAXSI) {
	  if (auto ft = llvm::dyn_cast<FloatType>(t))
	    identityAttr = rewriter.getFloatAttr(ft, -std::numeric_limits<double>::infinity());
	  else
	    identityAttr = rewriter.getIntegerAttr(t, std::numeric_limits<int64_t>::min());
	} else {
	  identityAttr = rewriter.getZeroAttr(t);
	}
	Value identity = rewriter.create<arith::ConstantOp>(loc, cast<TypedAttr>(identityAttr));
	vectorInits.push_back(rewriter.create<vector::BroadcastOp>(loc, VectorType::get({vWidth}, t), identity));
      }

      // First pass, check for FMAs
      llvm::DenseSet<Operation*> fusedMultiplies;
      llvm::DenseMap<Operation*, arith::MulFOp> fmaCandidates;

      for (Operation &innerOp : op.getBody()->getOperations()) {
	auto addOp = dyn_cast<arith::AddFOp>(innerOp);
	if (!addOp) continue;

	auto checkMul = [&](Value v) -> arith::MulFOp {
	  Operation *defOp = v.getDefiningOp();
	  if (!defOp || defOp == op || defOp->getName().getStringRef() != "arith.mulf") 
	    return nullptr;
	  if (!v.hasOneUse()) return nullptr; 
        
	  auto m = cast<arith::MulFOp>(defOp);
	  if (m->getBlock() == op.getBody()) return m;
	  return nullptr;
	};

	if (auto m = checkMul(addOp.getLhs())) {
	  fmaCandidates[addOp] = m;
	  fusedMultiplies.insert(m);
	} else if (auto m = checkMul(addOp.getRhs())) {
	  fmaCandidates[addOp] = m;
	  fusedMultiplies.insert(m);
	}
      }

      // Second pass, transform atomics
      auto newLoop = rewriter.create<scf::ParallelOp>(loc, op.getLowerBound(), op.getUpperBound(), ValueRange{stepConst}, vectorInits,
						      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange ivs, ValueRange /*iterArgs*/) {
        
        Value zeroIdx = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, 0);
	Value ub = op.getUpperBound()[0];
	Value cstZero = nestedBuilder.create<arith::ConstantOp>(nestedLoc, nestedBuilder.getZeroAttr(elemType));
        
	SmallVector<Value> partialSums = vectorInits;
	
	for (int u = 0; u < targetUnrollFactor; ++u) {
	  IRMapping mapping;
	  Value offset = nestedBuilder.create<arith::ConstantIndexOp>(nestedLoc, u * vWidth);
	  Value iv = nestedBuilder.create<arith::AddIOp>(nestedLoc, ivs[0], offset);
	  mapping.map(op.getInductionVars()[0], iv);
	  
	  Value diff = nestedBuilder.create<arith::SubIOp>(nestedLoc, ub, iv);
	  Value clamped = nestedBuilder.create<arith::MaxSIOp>(nestedLoc, diff, zeroIdx);
	  Value mask = nestedBuilder.create<vector::CreateMaskOp>(nestedLoc, maskType, clamped);
	  
	  for (Operation &innerOp : op.getBody()->getOperations()) {
	    
	    // Handle reductions
	    if (innerOp.hasTrait<OpTrait::IsTerminator>()) {
	      if (auto origReduce = dyn_cast<scf::ReduceOp>(innerOp)) {
		for (auto [i, operand] : llvm::enumerate(origReduce.getOperands())) {
		  Value val = mapping.lookupOrDefault(operand);
		  if (!llvm::isa<VectorType>(val.getType()))
		    val = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, val);
		  
		  if (llvm::isa<FloatType>(elemType)) {
		    partialSums[i] = (redKind == vector::CombiningKind::MUL) ?
		      nestedBuilder.create<arith::MulFOp>(nestedLoc, partialSums[i], val).getResult() :
		      nestedBuilder.create<arith::AddFOp>(nestedLoc, partialSums[i], val).getResult();
		  } else {
		    partialSums[i] = (redKind == vector::CombiningKind::MUL) ?
		      nestedBuilder.create<arith::MulIOp>(nestedLoc, partialSums[i], val).getResult() :
		      nestedBuilder.create<arith::AddIOp>(nestedLoc, partialSums[i], val).getResult();
		  }
		}
	      }
	      continue;
	    }
	    
	    // Skip fused multiplies
	    if (fusedMultiplies.contains(&innerOp)) continue;
	    
	    // Generate FMAs
	    if (auto addOp = dyn_cast<arith::AddFOp>(innerOp)) {
	      if (fmaCandidates.count(addOp)) {
		arith::MulFOp mulOp = fmaCandidates[addOp];
		Value vL = mapping.lookupOrDefault(mulOp.getLhs());
		Value vR = mapping.lookupOrDefault(mulOp.getRhs());
		Value acc = (addOp.getLhs() == mulOp.getResult()) ? addOp.getRhs() : addOp.getLhs();
		Value vAcc = mapping.lookupOrDefault(acc);
		
		auto ensureVec = [&](Value v) {
		  return llvm::isa<VectorType>(v.getType()) ? v :
		    nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, v).getResult();
		};
		
		Value fma = nestedBuilder.create<vector::FMAOp>(nestedLoc, 
								ensureVec(vL), ensureVec(vR), ensureVec(vAcc));
		mapping.map(addOp.getResult(), fma);
		continue; 
	      }
							    }
	    
	    // Vectorise loads
	    if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
	      Value memref = mapping.lookupOrDefault(load.getMemref());
	      if (!llvm::isa<MemRefType>(memref.getType())) continue;
	      
	      SmallVector<Value> idxs;
	      for (auto i : load.getIndices()) idxs.push_back(mapping.lookupOrDefault(i));
              
	      AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1, nestedBuilder.getContext());
	      	      
	      auto readOp = nestedBuilder.create<vector::TransferReadOp>(nestedLoc, vType, memref, idxs, AffineMapAttr::get(map), cstZero, mask, nestedBuilder.getBoolArrayAttr({false}));
	      mapping.map(load.getResult(), readOp.getResult());
	    }
	    // Vectorise stores
	    else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
	      Value memref = mapping.lookupOrDefault(store.getMemref());
	      if (!llvm::isa<MemRefType>(memref.getType())) continue;
	      
	      Value val = mapping.lookupOrDefault(store.getValueToStore());
	      if (!llvm::isa<VectorType>(val.getType()))
		val = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, val);
	      
	      SmallVector<Value> idxs;
	      for (auto i : store.getIndices()) idxs.push_back(mapping.lookupOrDefault(i));
              
	      AffineMap map = AffineMap::getMinorIdentityMap(llvm::cast<MemRefType>(memref.getType()).getRank(), 1, nestedBuilder.getContext());
	      
	      nestedBuilder.create<vector::TransferWriteOp>(nestedLoc, val, memref, idxs, AffineMapAttr::get(map), mask, nestedBuilder.getBoolArrayAttr({false}));
	    }
	    // General arith/math vectorisation
	    else if (innerOp.getDialect()->getNamespace() == "arith" || 
		     innerOp.getDialect()->getNamespace() == "math") {
	      SmallVector<Value> ops;
	      for (auto o : innerOp.getOperands()) {
		Value v = mapping.lookupOrDefault(o);
		if (!llvm::isa<VectorType>(v.getType())) 
		  v = nestedBuilder.create<vector::BroadcastOp>(nestedLoc, vType, v);
		ops.push_back(v);
	      }
	      OperationState state(nestedLoc, innerOp.getName().getStringRef());
	      state.addOperands(ops);
	      for (Type t : innerOp.getResultTypes()) state.addTypes(VectorType::get({vWidth}, t));
	      state.addAttributes(innerOp.getAttrs());
	      Operation *vOp = nestedBuilder.create(state);
	      for (unsigned i = 0; i < innerOp.getNumResults(); ++i) 
		mapping.map(innerOp.getResult(i), vOp->getResult(i));
	    }
	  }
	}
	
	// Deal with the loop terminator
	auto reduceOp = nestedBuilder.create<scf::ReduceOp>(nestedLoc, partialSums);
	for (unsigned i = 0; i < reduceOp.getNumRegions(); ++i) {
	  OpBuilder::InsertionGuard guard(nestedBuilder);
	  Region &region = reduceOp.getRegion(i);
	  
	  // Fallback safety just in case the region is empty
	  if (region.empty()) {
	    nestedBuilder.createBlock(&region);
	  }
	  Block &redBlock = region.front();
	  
	  // Fallback safety if the block was created without arguments
	  if (redBlock.getNumArguments() == 0) {
	    redBlock.addArgument(vType, nestedLoc);
	    redBlock.addArgument(vType, nestedLoc);
	  }
	  
	  nestedBuilder.setInsertionPointToEnd(&redBlock);
	  
	  Value lhs = redBlock.getArgument(0);
	  Value rhs = redBlock.getArgument(1);
	  Value combined;
	  
	  if (llvm::isa<FloatType>(elemType)) {
	    switch (redKind) {
	    case vector::CombiningKind::MUL: combined = nestedBuilder.create<arith::MulFOp>(nestedLoc, lhs, rhs).getResult(); break;
	    case vector::CombiningKind::MAXSI: combined = nestedBuilder.create<arith::MaximumFOp>(nestedLoc, lhs, rhs).getResult(); break;
	    case vector::CombiningKind::MINSI: combined = nestedBuilder.create<arith::MinimumFOp>(nestedLoc, lhs, rhs).getResult(); break;
	    default: combined = nestedBuilder.create<arith::AddFOp>(nestedLoc, lhs, rhs).getResult(); break;
	    }
	  } else {
	    switch (redKind) {
	    case vector::CombiningKind::MUL: combined = nestedBuilder.create<arith::MulIOp>(nestedLoc, lhs, rhs).getResult(); break;
							    case vector::CombiningKind::MAXSI: combined = nestedBuilder.create<arith::MaxSIOp>(nestedLoc, lhs, rhs).getResult(); break;
	    case vector::CombiningKind::MINSI: combined = nestedBuilder.create<arith::MinSIOp>(nestedLoc, lhs, rhs).getResult(); break;
	    default: combined = nestedBuilder.create<arith::AddIOp>(nestedLoc, lhs, rhs).getResult(); break;
	    }
	  }
	  nestedBuilder.create<scf::ReduceReturnOp>(nestedLoc, combined);
	}
      });
      
      // Mark the loop as vectorised so it doesn't get reprocessed
      newLoop->setAttr("vectorized", rewriter.getUnitAttr());
      rewriter.setInsertionPointAfter(newLoop);

      // Do the final reduction
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

      RewritePatternSet vectorPatterns(ctx);
      vectorPatterns.add<CUDAToVectorPattern>(ctx, finalWidth, finalUnrollFactor);
      FrozenRewritePatternSet frozenVectorPatterns(std::move(vectorPatterns));

      // Gather loops first to avoid "iterator invalidation" or seeing new loops
      SmallVector<scf::ParallelOp> leafLoops;
      module.walk([&](scf::ParallelOp op) {
        if (!op->hasAttr("vectorized") && !op->hasAttr("tiled")) {
	  bool hasNested = false;
	  op.getRegion().walk([&](scf::ParallelOp nested) {
	    if (nested != op) hasNested = true;
	  });
	  if (!hasNested) leafLoops.push_back(op);
        }
      });

      for (auto op : leafLoops) {
        // Wrap the single op in a brace-init list to satisfy the ArrayRef requirement
        (void)applyOpPatternsAndFold({op.getOperation()}, frozenVectorPatterns);
      }
    }
    
  };
  
}
