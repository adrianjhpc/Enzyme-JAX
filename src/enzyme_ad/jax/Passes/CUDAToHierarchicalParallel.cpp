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
    
      auto loc = op.getLoc();
    
      // Setup types and constants
      Type f32Type = rewriter.getF32Type();
      // Detect element type from the first operation in the body that has a type
      Type elementType = f32Type; // default
      for (auto &op : *op.getBody()) {
	if (auto load = dyn_cast<memref::LoadOp>(op)) {
	  elementType = mlir::cast<MemRefType>(load.getMemref().getType()).getElementType();
        break;
	}
      }
      unsigned bitWidth = elementType.getIntOrFloatBitWidth();      

      if (targetBitWidth % bitWidth != 0) {
	return rewriter.notifyMatchFailure(op, "Incompatible register/element width");
      }
      
      unsigned vWidth = targetBitWidth / bitWidth;

      
      VectorType vType = VectorType::get({vWidth}, f32Type);
      VectorType maskType = VectorType::get({vWidth}, rewriter.getI1Type());
    
      if (op.getStep().empty()) return failure();
      Value scalarIv = op.getInductionVars()[0];
      Value ub = op.getUpperBound()[0];
    
      // Update loop step to vWidth
      rewriter.modifyOpInPlace(op, [&]() {
	Value newStep = rewriter.create<arith::ConstantIndexOp>(loc, vWidth);
	op.getStepMutable().assign(newStep);
      });
    
      // Setup masking and thread id vectorisation
      rewriter.setInsertionPointToStart(op.getBody());
    
      // Create a thread id vector (i.e. [i, i+1, ..., i+7])
      SmallVector<float, 8> offsets;
      for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(static_cast<float>(i));
      auto offsetsConst = rewriter.create<arith::ConstantOp>(loc, vType, rewriter.getF32VectorAttr(offsets));
    
      Value floatIv = rewriter.create<arith::IndexCastOp>(loc, f32Type, scalarIv).getResult();
      Value splatIv = rewriter.create<vector::BroadcastOp>(loc, vType, floatIv).getResult();

      Value vectorisedThreadId = rewriter.create<arith::AddFOp>(loc, splatIv, offsetsConst).getResult();

    
      // This map tracks scalar values to their vectorised equivalents
      DenseMap<Value, Value> vectorisedMapping;
      vectorisedMapping[scalarIv] = vectorisedThreadId;
    
      // Standard attributes for TransferRead/Write
      auto mapAttr = AffineMapAttr::get(rewriter.getMultiDimIdentityMap(1));
      auto inBoundsAttr = rewriter.getBoolArrayAttr({false});

      //    Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, rewriter.create<arith::SubIOp>(loc, ub, scalarIv));

      Value diff = rewriter.create<arith::SubIOp>(loc, ub, scalarIv).getResult();

      Value mask = rewriter.create<vector::CreateMaskOp>(loc, maskType, diff);
      Value zeroPadding = rewriter.create<arith::ConstantOp>(loc, f32Type, rewriter.getF32FloatAttr(0.0f));

      // Deal with different types of scalar values
      auto getVecOp = [&](Value scalarVal) -> Value {
	if (vectorisedMapping.count(scalarVal))
	  return vectorisedMapping[scalarVal];
	
	auto type = scalarVal.getType();
	// Create a vector version of the scalar type
	auto vType = VectorType::get({vWidth}, type);

	rewriter.setInsertionPointAfterValue(scalarVal);
	// If it's a constant or a uniform value, splat/broadcast it
	return rewriter.create<vector::BroadcastOp>(loc, vType, scalarVal).getResult();
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
	// This handles AddF, SubF, MulF, DivF, AddI, SubI, MulI, AndI, ShlI, etc.
	else if (innerOp.getDialect()->getNamespace() == "arith") {
	  // Only handle standard element-wise ops (1 result, 1 or more operands)
	  if (innerOp.getNumResults() == 1 && innerOp.getNumOperands() >= 1) {
	    rewriter.setInsertionPoint(&innerOp);
	    
	    // Vectorise all operands (LHS, RHS, etc.)
	    SmallVector<Value, 2> vOperands;
	    for (Value operand : innerOp.getOperands()) {
	      vOperands.push_back(getVecOp(operand));
	    }
	    
	    // Determine the new vector result type
	    Type scalarRetType = innerOp.getResult(0).getType();
	    VectorType vRetType = VectorType::get({vWidth}, scalarRetType);
	    
	    // Create a new version of the op with vector operands and result
	    // OperationState allows us to create an op by name (e.g., "arith.addf")
	    OperationState state(loc, innerOp.getName().getStringRef());
	    state.addOperands(vOperands);
	    state.addTypes(vRetType);
	    state.addAttributes(innerOp.getAttrs());
	    
	    Operation *vOp = rewriter.create(state);
	    
	    // Update mapping and replace
	    vectorisedMapping[innerOp.getResult(0)] = vOp->getResult(0);
	    rewriter.replaceOp(&innerOp, vOp->getResults());
	    continue; // Skip to next iteration
	  }
	}

	// --- MATH OPERATIONS ---
	else if (mlir::isa<math::ExpOp, math::SqrtOp, math::TanhOp, math::SinOp, math::CosOp>(innerOp)) {
	
	  rewriter.setInsertionPoint(&innerOp);
	
	  // Get the vectorised version of the input operand
	  Value scalarInput = innerOp.getOperand(0);
	  Value vInput = getVecOp(scalarInput);
	
	  Operation *vMath = nullptr;
	
	  // Create the vector-version of the math op
	  if (mlir::isa<math::ExpOp>(innerOp))       vMath = rewriter.create<math::ExpOp>(loc, vInput);
	  else if (mlir::isa<math::SinOp>(innerOp))  vMath = rewriter.create<math::SinOp>(loc, vInput);
	  else if (mlir::isa<math::CosOp>(innerOp))  vMath = rewriter.create<math::CosOp>(loc, vInput);
	  else if (mlir::isa<math::TanhOp>(innerOp)) vMath = rewriter.create<math::TanhOp>(loc, vInput);
	  else if (mlir::isa<math::LogOp>(innerOp))  vMath = rewriter.create<math::LogOp>(loc, vInput);
	  else if (mlir::isa<math::SqrtOp>(innerOp)) vMath = rewriter.create<math::SqrtOp>(loc, vInput);
	
	  if (vMath) {
	    // Register the result so subsequent ops can use it
	    vectorisedMapping[innerOp.getResult(0)] = vMath->getResult(0);
	    rewriter.replaceOp(&innerOp, vMath->getResults());
	  }
	}
	
	// --- SHUFFLE ---
	else if (auto shfl = dyn_cast<gpu::ShuffleOp>(innerOp)) {
	  rewriter.setInsertionPoint(shfl);
	  Value vInput = getVecOp(shfl.getValue());
	  uint32_t delta = 0;
	  if (auto constOp = shfl.getOffset().getDefiningOp<arith::ConstantIntOp>())
	    delta = constOp.value();
	
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

	// --- BARRIER ---
	else if (isa<gpu::BarrierOp>(innerOp)) {
	  rewriter.eraseOp(&innerOp);
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
	  Value vVal = getVecOp(atomic.getValue());
	  rewriter.setInsertionPoint(atomic);
	  auto reduction = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, vVal);
	  rewriter.create<memref::AtomicRMWOp>(
					       loc, arith::AtomicRMWKind::addf, reduction.getResult(), atomic.getMemref(), atomic.getIndices());
	  rewriter.eraseOp(atomic);
	}
      
      
	else if (auto call = dyn_cast<func::CallOp>(innerOp)) {
	  auto callee = call.getCallee();
	  if (callee == "sinf" || callee == "__nv_sinf") {
	    rewriter.setInsertionPoint(call);
	    Value vInput = getVecOp(call.getOperand(0));
	    auto vSin = rewriter.create<math::SinOp>(loc, vInput);
	    vectorisedMapping[call.getResult(0)] = vSin.getResult();
	    rewriter.replaceOp(call, vSin.getResult());
	  }else{
	    std::cout << "Error 'function call' identified but not handled: " << callee.str() << std::endl;
	  }
	}
      
      }
    
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
      module.walk([&](gpu::AllocOp alloc) {
	if (auto intAttr = mlir::dyn_cast_or_null<IntegerAttr>(alloc.getType().getMemorySpace())) {
	  if (intAttr.getInt() == 3) {
	    OpBuilder b(alloc);
	    auto stackMem = b.create<memref::AllocaOp>(alloc.getLoc(), mlir::cast<MemRefType>(alloc.getType()));
	    alloc->getResult(0).replaceAllUsesWith(stackMem->getResult(0));
	    alloc.erase();
	  }
	}
      });

      // Apply vectorisation to innermost (thread) loops
      RewritePatternSet patterns(ctx);
      patterns.add<CUDAToVectorPattern>(ctx, finalWidth);
      if (failed(applyPatternsGreedily(module, std::move(patterns))))
	signalPassFailure();
    }
  };
    
}
