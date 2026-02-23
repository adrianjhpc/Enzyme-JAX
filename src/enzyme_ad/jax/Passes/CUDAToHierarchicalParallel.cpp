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

  struct CUDAToVectorPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;
    
    int targetBitWidth;
    
    CUDAToVectorPattern(MLIRContext *context, int bitWidth)
      : OpRewritePattern<scf::ParallelOp>(context), targetBitWidth(bitWidth) {}
  
    LogicalResult matchAndRewrite(scf::ParallelOp op, PatternRewriter &rewriter) const override {

      // Safety Check: Ensure it's a 1D parallel loop before splitting
      if (op.getInductionVars().size() != 1) return failure();

      auto loc = op.getLoc();
      
      // --- BUG FIX: Dynamic Element Type Detection ---
      Type elementType = rewriter.getF32Type(); // default fallback
      for (auto &innerOp : *op.getBody()) {
        if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
          elementType = mlir::cast<MemRefType>(load.getMemref().getType()).getElementType();
          break;
        } else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
          elementType = mlir::cast<MemRefType>(store.getMemref().getType()).getElementType();
          break;
        }
      }
      
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();      
      if (targetBitWidth % bitWidth != 0) {
        return rewriter.notifyMatchFailure(op, "Incompatible register/element width");
      }
      
      unsigned vWidth = targetBitWidth / bitWidth;
      
      VectorType vType = VectorType::get({vWidth}, elementType);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
    
      if (op.getStep().empty()) return failure();
      Value scalarIv = op.getInductionVars()[0];
      Value ub = op.getUpperBound()[0];
    
      // Update loop step to vWidth
      rewriter.modifyOpInPlace(op, [&]() {
        Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
        op.getStepMutable().assign(newStep);
      });
    
      rewriter.setInsertionPointToStart(op.getBody());
    
      // --- BUG FIX: Thread ID Vectorization with Dynamic Types ---
      Value baseThreadId;
      if (isa<FloatType>(elementType)) {
        // Cast index -> i32 -> float
        Value intIv = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), scalarIv);
        baseThreadId = rewriter.create<arith::SIToFPOp>(loc, elementType, intIv);
      } else {
        // Cast index -> int directly
        baseThreadId = rewriter.create<arith::IndexCastOp>(loc, elementType, scalarIv);
      }
      Value splatIv = rewriter.create<vector::BroadcastOp>(loc, vType, baseThreadId).getResult();

      // Dynamically create sequence offsets [0, 1, 2, ... vWidth-1]
      Attribute offsetsAttr;
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        SmallVector<APFloat, 8> offsets;
        for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(APFloat(floatType.getFloatSemantics(), i));
        offsetsAttr = DenseElementsAttr::get(vType, offsets);
      } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
        SmallVector<APInt, 8> offsets;
        for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(APInt(intType.getWidth(), i));
        offsetsAttr = DenseElementsAttr::get(vType, offsets);
      }
      Value offsetsConst = rewriter.create<arith::ConstantOp>(loc, vType, offsetsAttr);
    
      Value vectorisedThreadId;
      if (isa<FloatType>(elementType)) {
        vectorisedThreadId = rewriter.create<arith::AddFOp>(loc, splatIv, offsetsConst);
      } else {
        vectorisedThreadId = rewriter.create<arith::AddIOp>(loc, splatIv, offsetsConst);
      }

      DenseMap<Value, Value> vectorisedMapping;
      vectorisedMapping[scalarIv] = vectorisedThreadId;
    
      auto mapAttr = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));
      auto inBoundsAttr = rewriter.getBoolArrayAttr({false});

      Value diff = rewriter.create<arith::SubIOp>(loc, ub, scalarIv).getResult();
      Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);

      // --- BUG FIX: Dynamic Zero Padding ---
      Value zeroPadding;
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
          zeroPadding = rewriter.create<arith::ConstantOp>(loc, floatType, rewriter.getFloatAttr(floatType, 0.0));
      } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
          zeroPadding = rewriter.create<arith::ConstantIntOp>(loc, 0, intType.getWidth());
      }    

      // Deal with different types of scalar values
      auto getVecOp = [&](Value scalarVal) -> Value {
        if (vectorisedMapping.count(scalarVal))
          return vectorisedMapping[scalarVal];
        
        auto type = scalarVal.getType();
        auto vType = VectorType::get({vWidth}, type);

        // --- BUG FIX: Protect the Rewriter's Insertion Point ---
        OpBuilder::InsertionGuard guard(rewriter);
        
        rewriter.setInsertionPointAfterValue(scalarVal);
        Value splat = rewriter.create<vector::BroadcastOp>(loc, vType, scalarVal).getResult();
        
        vectorisedMapping[scalarVal] = splat;
        return splat;
      }; 
      // Transform the body
      for (Operation &innerOp : llvm::make_early_inc_range(*op.getBody())) {
      
	// --- LOAD ---
	if (auto load = dyn_cast<memref::LoadOp>(innerOp)) {
	  rewriter.setInsertionPoint(load);
	  SmallVector<Value, 4> indices(load.getIndices());
	  auto vRead = rewriter.create<vector::TransferReadOp>(
							       loc, vType, load.getMemref(), indices, mapAttr, zeroPadding, mask, inBoundsAttr);
	
	  vectorisedMapping[load.getResult()] = vRead.getResult(); // Record result
	  rewriter.replaceOp(load, vRead.getResult());
	}
      

        // --- GENERIC ARITHMETIC HANDLER ---
        else if (innerOp.getDialect()->getNamespace() == "arith") {
          if (innerOp.getNumResults() == 1 && innerOp.getNumOperands() >= 1) {
            rewriter.setInsertionPoint(&innerOp);
            
            SmallVector<Value, 2> vOperands;
            for (auto [idx, operand] : llvm::enumerate(innerOp.getOperands())) {
              Value vOp = getVecOp(operand);
              
              // --- TAIL MASKING FIX: Protect Division by Zero ---
              // If this is the right-hand side (idx == 1) of a division, prevent X / 0.0 on padding lanes
              if (idx == 1 && mlir::isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(innerOp)) {
                 Type opElemType = mlir::cast<VectorType>(vOp.getType()).getElementType();
                 Value safeScalar;
                 if (auto fType = dyn_cast<FloatType>(opElemType)) {
                     safeScalar = rewriter.create<arith::ConstantOp>(loc, fType, rewriter.getFloatAttr(fType, 1.0));
                 } else if (auto iType = dyn_cast<IntegerType>(opElemType)) {
                     safeScalar = rewriter.create<arith::ConstantIntOp>(loc, 1, iType.getWidth());
                 }
                 VectorType vSafePadType = VectorType::get({vWidth}, opElemType);
                 Value vSafePad = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalar);
                 
                 // Swap 0.0 to 1.0 for out-of-bounds division
		 vOp = rewriter.create<arith::SelectOp>(loc, mask, vOp, vSafePad).getResult();
              }
              vOperands.push_back(vOp);
            }
            
            Type scalarRetType = innerOp.getResult(0).getType();
            VectorType vRetType = VectorType::get({vWidth}, scalarRetType);
            
            OperationState state(loc, innerOp.getName().getStringRef());
            state.addOperands(vOperands);
            state.addTypes(vRetType);
            state.addAttributes(innerOp.getAttrs());
            
            Operation *vOp = rewriter.create(state);
            vectorisedMapping[innerOp.getResult(0)] = vOp->getResult(0);
            rewriter.replaceOp(&innerOp, vOp->getResults());
            continue;
          }
        }

        // --- MATH OPERATIONS ---
        else if (mlir::isa<math::ExpOp, math::SqrtOp, math::RsqrtOp, math::TanhOp, math::ErfOp, math::SinOp, math::CosOp, math::LogOp, math::PowFOp>(innerOp)) {
     	  rewriter.setInsertionPoint(&innerOp);
        
          Value scalarInput = innerOp.getOperand(0);
          Value vInput = getVecOp(scalarInput);
        
          // --- TAIL MASKING FIX: Protect against NaN/Inf FPEs in padded lanes ---
          Type opElemType = mlir::cast<VectorType>(vInput.getType()).getElementType();
          Value safeScalarPad;
          
          // Provide 1.0 for Log/Sqrt to prevent evaluating Log(0.0)= -Inf or Sqrt(-x)= NaN. 
          // Otherwise default to 0.0.
          if (mlir::isa<math::LogOp, math::SqrtOp>(innerOp)) {
            safeScalarPad = rewriter.create<arith::ConstantOp>(loc, opElemType, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 1.0));
          } else {
            safeScalarPad = rewriter.create<arith::ConstantOp>(loc, opElemType, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 0.0));
          }
          
          VectorType vSafePadType = VectorType::get({vWidth}, opElemType);
          Value vSafePad = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalarPad);
          
          // Select safe values for inactive tail lanes
          Value safeInput = rewriter.create<arith::SelectOp>(loc, mask, vInput, vSafePad).getResult();
 
          Operation *vMath = nullptr;
          if (mlir::isa<math::ExpOp>(innerOp))       vMath = rewriter.create<math::ExpOp>(loc, safeInput);
          else if (mlir::isa<math::SinOp>(innerOp))  vMath = rewriter.create<math::SinOp>(loc, safeInput);
          else if (mlir::isa<math::CosOp>(innerOp))  vMath = rewriter.create<math::CosOp>(loc, safeInput);
          else if (mlir::isa<math::TanhOp>(innerOp)) vMath = rewriter.create<math::TanhOp>(loc, safeInput);
          else if (mlir::isa<math::LogOp>(innerOp))  vMath = rewriter.create<math::LogOp>(loc, safeInput);
          else if (mlir::isa<math::SqrtOp>(innerOp)) vMath = rewriter.create<math::SqrtOp>(loc, safeInput);
          else if (mlir::isa<math::RsqrtOp>(innerOp)) vMath = rewriter.create<math::RsqrtOp>(loc, safeInput);
          else if (mlir::isa<math::ErfOp>(innerOp))   vMath = rewriter.create<math::ErfOp>(loc, safeInput);
	  else if (mlir::isa<math::PowFOp>(innerOp)) {

            auto powf = dyn_cast<math::PowFOp>(innerOp);
            rewriter.setInsertionPoint(powf);
        
            // Extract both the Base and the Exponent
            Value scalarBase = powf.getOperand(0);
            Value scalarExp = powf.getOperand(1);
            Value vBase = getVecOp(scalarBase);
            Value vExp = getVecOp(scalarExp);
        
            // --- TAIL MASKING FIX: Protect against NaN/Inf FPEs ---
            Type opElemType = mlir::cast<VectorType>(vBase.getType()).getElementType();
          
            // Safe padding for powf: 1.0 ^ 1.0 = 1.0 (safe, no FPEs)
            Value safeScalarPad = rewriter.create<arith::ConstantOp>(
                loc, opElemType, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 1.0));
          
            VectorType vSafePadType = VectorType::get({vWidth}, opElemType);
            Value vSafePad = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalarPad).getResult();
          
            // Select safe values for inactive tail lanes for BOTH operands
            Value safeBase = rewriter.create<arith::SelectOp>(loc, mask, vBase, vSafePad).getResult();
            Value safeExp = rewriter.create<arith::SelectOp>(loc, mask, vExp, vSafePad).getResult();
        
            // Create the vectorised PowFOp
            auto vPowf = rewriter.create<math::PowFOp>(loc, safeBase, safeExp);
        
            // Register the result and replace the original operation
            vectorisedMapping[powf.getResult()] = vPowf.getResult();
            rewriter.replaceOp(powf, vPowf.getResult());
          }

	  if (vMath) {
            vectorisedMapping[innerOp.getResult(0)] = vMath->getResult(0);
            rewriter.replaceOp(&innerOp, vMath->getResults());
          }
        }	
	
	// --- SHUFFLE ---
	else if (auto shfl = dyn_cast<gpu::ShuffleOp>(innerOp)) {
	  rewriter.setInsertionPoint(shfl);
	  Value vInput = getVecOp(shfl.getValue());
	  auto constOp = shfl.getOffset().getDefiningOp<arith::ConstantIntOp>();
          if (!constOp) {
            // Cannot handle dynamic shuffle offsets in pure static vector IR easily.
            return rewriter.notifyMatchFailure(shfl, "Dynamic shuffle offset not supported");
          }
          uint32_t delta = constOp.value();
	
	  if (shfl.getMode() == gpu::ShuffleMode::DOWN) {
	    SmallVector<int64_t> shuffleMask;
	    for (int64_t i = 0; i < vWidth; ++i){
	      shuffleMask.push_back(std::min<int64_t>(i + delta, vWidth - 1));
	    }	  
	    auto vShfl = rewriter.create<vector::ShuffleOp>(loc, vInput, vInput, shuffleMask);
	    vectorisedMapping[shfl.getResult(0)] = vShfl.getResult();
	    rewriter.replaceOp(shfl, vShfl.getResult());
	  }

	  else if (shfl.getMode() == gpu::ShuffleMode::XOR) {
	    // [0,1,2,3] XOR 1 -> [1,0,3,2]
	    SmallVector<int64_t> xorMask;
	    for (int64_t i = 0; i < vWidth; ++i) {
	      xorMask.push_back(i ^ delta);
	    }
	    auto vShfl = rewriter.create<vector::ShuffleOp>(loc, vInput, vInput, xorMask);
	    vectorisedMapping[shfl.getResult(0)] = vShfl.getResult();
	    rewriter.replaceOp(shfl, vShfl.getResult());
	  }
	
	}

        // --- SUBGROUP / WARP REDUCTION ---
        else if (auto sgReduce = dyn_cast<gpu::SubgroupReduceOp>(innerOp)) {
          rewriter.setInsertionPoint(sgReduce);
          Value vInput = getVecOp(sgReduce.getValue());
          
          vector::CombiningKind kind;
          switch (sgReduce.getOp()) {
            case gpu::AllReduceOperation::ADD: kind = vector::CombiningKind::ADD; break;
            case gpu::AllReduceOperation::MUL: kind = vector::CombiningKind::MUL; break;
            case gpu::AllReduceOperation::MINNUM: kind = vector::CombiningKind::MINF; break;
            case gpu::AllReduceOperation::MAXNUM: kind = vector::CombiningKind::MAXF; break;
            // Add integer cases (MINUI, MINSI) as needed
            default: return rewriter.notifyMatchFailure(sgReduce, "Unsupported reduction kind");
          }

          // Reduce the vector into a scalar
          Value scalarReduction = rewriter.create<vector::ReductionOp>(loc, kind, vInput).getResult();
          
          // A subgroup reduction returns the scalar result to ALL threads, 
          // so broadcast it back out to the vector
          Value vBroadcastBack = rewriter.create<vector::BroadcastOp>(loc, vType, scalarReduction).getResult();
          
          vectorisedMapping[sgReduce.getResult()] = vBroadcastBack;
          rewriter.replaceOp(sgReduce, vBroadcastBack);
        }

	// --- STORE ---
	else if (auto store = dyn_cast<memref::StoreOp>(innerOp)) {
	  rewriter.setInsertionPoint(store);
	  Value vVal = getVecOp(store.getValueToStore());
	  SmallVector<Value, 4> indices(store.getIndices());
        
	  rewriter.create<vector::TransferWriteOp>(
						   loc, vVal, store.getMemref(), indices, mapAttr, mask, inBoundsAttr);
	  rewriter.eraseOp(store);
	}
      
	// --- ATOMIC ADD ---
        else if (auto atomic = dyn_cast<memref::AtomicRMWOp>(innerOp)) {
          rewriter.setInsertionPoint(atomic);
          Value vVal = getVecOp(atomic.getValue());

          // 1. Detect if the target address is Uniform or Varying (Scatter)
          bool isUniformAddress = true;
          for (Value idx : atomic.getIndices()) {
            // If an index was vectorised, it means threads write to different locations
            if (vectorisedMapping.count(idx)) {
              isUniformAddress = false;
              break;
            }
          }

          if (isUniformAddress) {
            // UNIFORM: All lanes add to the exact same memory address.
            // Safely mask inactive padding lanes by replacing them with 0.0
            Value neutralElement = rewriter.create<arith::ConstantOp>(loc, vType, rewriter.getZeroAttr(vType));
            
            // Uniform Select:
            Value maskedVal = rewriter.create<arith::SelectOp>(loc, mask, vVal, neutralElement).getResult();

            // Reduce the vector into a scalar
            auto reduction = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, maskedVal);
            
            // Issue a single scalar atomic
            rewriter.create<memref::AtomicRMWOp>(loc, atomic.getKind(), reduction.getResult(), atomic.getMemref(), atomic.getIndices());


	  } else {
            // VARYING (SCATTER): Lanes write to different addresses.
            // MLIR vector lacks a universal scatter-atomic, so we unroll over the vector width.
            
            // Ensure all indices are available as vectors
            SmallVector<Value, 4> vIndices;
            for (Value idx : atomic.getIndices()) {
              vIndices.push_back(getVecOp(idx));
            }

            // Unroll over the vector width
            for (int64_t i = 0; i < vWidth; ++i) {
              // Extract the mask bit for this specific lane
              Value laneMask = rewriter.create<vector::ExtractOp>(loc, mask, ArrayRef<int64_t>{i}).getResult();
              
              // Conditionally execute the atomic using scf.if
              rewriter.create<scf::IfOp>(loc, laneMask, [&](OpBuilder &b, Location l) {
                // Extract the value to add
                Value laneVal = b.create<vector::ExtractOp>(l, vVal, ArrayRef<int64_t>{i}).getResult();
                
                // Extract the specific memory indices for this lane
                SmallVector<Value, 4> laneIndices;
                for (Value vIdx : vIndices) {
                  laneIndices.push_back(b.create<vector::ExtractOp>(l, vIdx, ArrayRef<int64_t>{i}).getResult());
                }
                
                // Issue the scalar atomic for this lane
                b.create<memref::AtomicRMWOp>(l, atomic.getKind(), laneVal, atomic.getMemref(), laneIndices);
                b.create<scf::YieldOp>(l);
              });
            }
          }
          rewriter.eraseOp(atomic);
        }
      
      }
    
      return success();
    }

  };


struct BarrierFissionPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp parallelOp, PatternRewriter &rewriter) const override {
      // 1. Find the first barrier in the loop
      gpu::BarrierOp barrier;
      for (auto &op : *parallelOp.getBody()) {
        if (auto b = dyn_cast<gpu::BarrierOp>(&op)) {
          barrier = b;
          break;
        }
      }
      if (!barrier) return failure(); // No barrier found, nothing to do

      Location loc = parallelOp.getLoc();
      
      // We assume a 1D parallel loop for simplicity (standard for threadIdx.x flattening)
      Value lb = parallelOp.getLowerBound()[0];
      Value ub = parallelOp.getUpperBound()[0];
      Value step = parallelOp.getStep()[0];
      Value iv1 = parallelOp.getInductionVars()[0];

      // 2. Identify Scalar Expansion candidates (values defined before barrier, used after)
      SmallVector<Value> crossBarrierValues;
      for (Operation *op = &parallelOp.getBody()->front(); op != barrier.getOperation(); op = op->getNextNode()) {
        for (Value result : op->getResults()) {
          bool usedAfter = false;
          for (Operation *user : result.getUsers()) {
            if (barrier->isBeforeInBlock(user)) {
              usedAfter = true;
              break;
            }
          }
          if (usedAfter) crossBarrierValues.push_back(result);
        }
      }

      // 3. Allocate thread-local memory to safely pass values across the barrier boundary
      rewriter.setInsertionPoint(parallelOp);
      DenseMap<Value, Value> allocations;
      for (Value val : crossBarrierValues) {
        MemRefType allocType = MemRefType::get({ShapedType::kDynamic}, val.getType());
        Value alloc = rewriter.create<memref::AllocaOp>(loc, allocType, ub);
        allocations[val] = alloc;
      }

      // 4. Create the second half of the loop
      rewriter.setInsertionPointAfter(parallelOp);
      auto secondLoop = rewriter.create<scf::ParallelOp>(loc, parallelOp.getLowerBound(), parallelOp.getUpperBound(), parallelOp.getStep());
      Value iv2 = secondLoop.getInductionVars()[0];

      // 5. Insert Stores in the First Loop (Right before the barrier)
      rewriter.setInsertionPoint(barrier);
      for (Value val : crossBarrierValues) {
        rewriter.create<memref::StoreOp>(loc, val, allocations[val], ValueRange{iv1});
      }

      // 6. Insert Loads in the Second Loop
      rewriter.setInsertionPointToStart(secondLoop.getBody());
      DenseMap<Value, Value> loads;
      for (Value val : crossBarrierValues) {
        loads[val] = rewriter.create<memref::LoadOp>(loc, allocations[val], ValueRange{iv2}).getResult();
      }

      // 7. Move all operations after the barrier into the second loop
      Operation *curr = barrier->getNextNode();
      Operation *end = &parallelOp.getBody()->back(); // scf.yield
      while (curr != end) {
        Operation *next = curr->getNextNode();
        curr->moveBefore(secondLoop.getBody()->getTerminator());
        
        // Deeply replace uses of the old variables and induction var with the loaded versions
        curr->walk([&](Operation *nestedOp) {
          for (OpOperand &operand : nestedOp->getOpOperands()) {
            if (loads.count(operand.get())) {
              operand.set(loads[operand.get()]);
            } else if (operand.get() == iv1) {
              operand.set(iv2); // Swap induction variables
            }
          }
        });
        curr = next;
      }

      // 8. Erase the old barrier
      rewriter.eraseOp(barrier);

      return success();
    }
  };

  struct CUDAToHierarchicalParallelPass : public enzyme::impl::CUDAToHierarchicalParallelBase<CUDAToHierarchicalParallelPass>  {
  using Base::Base;
  
    void runOnOperation() override {
      ModuleOp module = getOperation();
      MLIRContext *ctx = &getContext();
 
      assert(regBitWidth > 0 && "expected positive bit width for vectorisation");
      assert((regBitWidth & (regBitWidth - 1)) == 0 && "expected the bit width for vectorisation to be a power of 2");

      int finalWidth = regBitWidth; // Start with the default/CLI value

      // Check if the module has target features (often added by frontends or previous passes)
      if (auto attr = module->getAttrOfType<StringAttr>("llvm.target_features")) {
        llvm::StringRef features = attr.getValue();
        if (features.contains("+avx512f")) {
	  finalWidth = 512;
        } else if (features.contains("+avx2") || features.contains("+avx")) {
	  finalWidth = 256;
        } else if (features.contains("+neon") || features.contains("+sse")) {
	  finalWidth = 128;
        }
      }

      // Move shared memory (CUDA) to stack (Alloca)
      // We place it at the top of the block parallel loop (grid loop body)
      // We collect them first to avoid iterator invalidation during mutation
      SmallVector<gpu::AllocOp> allocsToMove;
      module.walk([&](gpu::AllocOp alloc) {
        if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(alloc.getType().getMemorySpace())) {
          if (intAttr.getInt() == 3) {
            allocsToMove.push_back(alloc);
          }
        }
      });

      for (auto alloc : allocsToMove) {
        func::FuncOp parentFunc = alloc->getParentOfType<func::FuncOp>();
        if (!parentFunc) continue; // Safety check
        OpBuilder hoistedBuilder(&parentFunc.getBody().front(), parentFunc.getBody().front().begin());
        auto stackMem = hoistedBuilder.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
        alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
        alloc.erase();
      }
      // Apply Barrier Fission to split loops securely before vectorisation happens
      RewritePatternSet prePatterns(ctx);
      prePatterns.add<BarrierFissionPattern>(ctx);
      if (failed(applyPatternsGreedily(module, std::move(prePatterns)))) {
        signalPassFailure();
      }

      // Apply vectorisation to innermost (thread) loops
      RewritePatternSet patterns(ctx);
      patterns.add<CUDAToVectorPattern>(ctx, finalWidth);
      if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
        signalPassFailure();
      }

    }
  };
    
}
