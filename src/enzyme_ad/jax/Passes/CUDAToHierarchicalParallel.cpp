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
      if (numDims <= 1) return failure(); // Already 1D, nothing to do.

      Location loc = op.getLoc();
      Value one = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
      Value zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));

      Value totalSize = one;
      SmallVector<Value> sizes;

      // Calculate the total 1D size of the flattened iteration space
      for (unsigned i = 0; i < numDims; ++i) {
        Value lb = op.getLowerBound()[i];
        Value ub = op.getUpperBound()[i];
        Value step = op.getStep()[i];

        // size[i] = (ub - lb + step - 1) / step
        Value diff = rewriter.create<arith::SubIOp>(loc, ub, lb);
        Value stepMinusOne = rewriter.create<arith::SubIOp>(loc, step, one);
        Value diffPlus = rewriter.create<arith::AddIOp>(loc, diff, stepMinusOne);
        Value size = rewriter.create<arith::DivUIOp>(loc, diffPlus, step);
        
        sizes.push_back(size);
        totalSize = rewriter.create<arith::MulIOp>(loc, totalSize, size);
      }

      // Create the new 1D parallel loop
      auto newLoop = rewriter.create<scf::ParallelOp>(
          loc, ValueRange{zero}, ValueRange{totalSize}, ValueRange{one}, op.getInitVals());
      
      rewriter.setInsertionPointToStart(newLoop.getBody());
      Value currentVal = newLoop.getInductionVars()[0];
      SmallVector<Value> decodedIvs(numDims);

      // Decode the 1D index back into 2D/3D indices for the loop body to use
      for (int i = numDims - 1; i >= 0; --i) {
        Value rem = rewriter.create<arith::RemUIOp>(loc, currentVal, sizes[i]);
        Value scaled = rewriter.create<arith::MulIOp>(loc, rem, op.getStep()[i]);
        decodedIvs[i] = rewriter.create<arith::AddIOp>(loc, op.getLowerBound()[i], scaled);
        if (i > 0) {
          currentVal = rewriter.create<arith::DivUIOp>(loc, currentVal, sizes[i]);
        }
      }

      // Splice the old loop's body into the new loop
      auto &oldBody = op.getBody()->getOperations();
      auto &newBody = newLoop.getBody()->getOperations();
      newBody.splice(std::prev(newBody.end()), oldBody, oldBody.begin(), std::prev(oldBody.end()));

      // Replace references to the old multi-dimensional indices with our decoded ones
      for (unsigned i = 0; i < numDims; ++i) {
        op.getInductionVars()[i].replaceAllUsesWith(decodedIvs[i]);
      }

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

      // Strip-Mining / Hierarchical Partitioning
      // We only vectorise innermost loops. 
      // If a loop contains another loop, it represents the Grid (CPU Threads).
      // We leave Grid loops untouched so they can map cleanly to OpenMP later.
      bool hasNestedParallel = false;
      op.walk([&](scf::ParallelOp nested) {
        if (nested != op) hasNestedParallel = true;
      });
      if (hasNestedParallel) return failure();

      // Safety Check: Ensure it's a 1D parallel loop before splitting
      if (op.getInductionVars().size() != 1) return failure();

      auto loc = op.getLoc();
      
      // Dynamic Element Type Detection
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
    
      // Update loop step to vWidth. 
      // (This inherently enables threading on flat 1D loops via strip-mining)
      rewriter.modifyOpInPlace(op, [&]() {
        Value newStep = rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(vWidth));
        op.getStepMutable().assign(newStep);
      });
    
      rewriter.setInsertionPointToStart(op.getBody());
    
      // Thread ID Vectorisation with Dynamic Types
      Value baseThreadId;
      if (isa<FloatType>(elementType)) {
        Value intIv = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), scalarIv);
        baseThreadId = rewriter.create<arith::SIToFPOp>(loc, elementType, intIv);
      } else {
        baseThreadId = rewriter.create<arith::IndexCastOp>(loc, elementType, scalarIv);
      }
      Value splatIv = rewriter.create<vector::BroadcastOp>(loc, vType, baseThreadId).getResult();

      DenseElementsAttr offsetsAttr; 
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        SmallVector<APFloat, 8> offsets;
        for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(APFloat(floatType.getFloatSemantics(), i));
        offsetsAttr = DenseElementsAttr::get(vType, offsets);
      } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
        SmallVector<APInt, 8> offsets;
        for (int64_t i = 0; i < vWidth; ++i) offsets.push_back(APInt(intType.getWidth(), i));
        offsetsAttr = DenseElementsAttr::get(vType, offsets);
      }
      Value offsetsConst = rewriter.create<arith::ConstantOp>(loc, offsetsAttr);
      
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

      // Dynamic Zero Padding
      Value zeroPadding;
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
          zeroPadding = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(floatType, 0.0));
      } else if (auto intType = dyn_cast<IntegerType>(elementType)) {
          zeroPadding = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(intType, 0));
      }

      // Deal with different types of scalar values
      auto getVecOp = [&](Value scalarVal) -> Value {
        if (vectorisedMapping.count(scalarVal))
          return vectorisedMapping[scalarVal];
        
        auto type = scalarVal.getType();
        auto vType = VectorType::get({vWidth}, type);

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
        
          vectorisedMapping[load.getResult()] = vRead.getResult(); 
          rewriter.replaceOp(load, vRead.getResult());
        }
      
        // --- GENERIC ARITHMETIC HANDLER ---
        else if (innerOp.getDialect()->getNamespace() == "arith") {
          if (innerOp.getNumResults() == 1 && innerOp.getNumOperands() >= 1) {
            rewriter.setInsertionPoint(&innerOp);
            
            SmallVector<Value, 2> vOperands;
            for (auto [idx, operand] : llvm::enumerate(innerOp.getOperands())) {
              Value vOp = getVecOp(operand);
              
              if (idx == 1 && mlir::isa<arith::DivFOp, arith::DivSIOp, arith::DivUIOp>(innerOp)) {
                 Type opElemType = mlir::cast<VectorType>(vOp.getType()).getElementType();
                 Value safeScalar;
                 if (auto fType = dyn_cast<FloatType>(opElemType)) {
                     safeScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(fType, 1.0));
                 } else if (auto iType = dyn_cast<IntegerType>(opElemType)) {
                     safeScalar = rewriter.create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(iType, 1));
                 }
                 
                 VectorType vSafePadType = VectorType::get({vWidth}, opElemType);
                 Value vSafePad = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalar);
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
        
          Type opElemType = mlir::cast<VectorType>(vInput.getType()).getElementType();
          Value safeScalarPad;
          
          if (mlir::isa<math::LogOp, math::SqrtOp>(innerOp)) {
            safeScalarPad = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 1.0));
          } else {
            safeScalarPad = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 0.0));
          }
          
          VectorType vSafePadType = VectorType::get({vWidth}, opElemType);
          Value vSafePad = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalarPad);
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
            Value scalarExp = powf.getOperand(1);
            Value vExp = getVecOp(scalarExp);
        
            Value safeScalarPadPow = rewriter.create<arith::ConstantOp>(loc, rewriter.getFloatAttr(mlir::cast<FloatType>(opElemType), 1.0));
            Value vSafePadPow = rewriter.create<vector::BroadcastOp>(loc, vSafePadType, safeScalarPadPow).getResult();
          
            Value safeBase = rewriter.create<arith::SelectOp>(loc, mask, vInput, vSafePadPow).getResult();
            Value safeExp = rewriter.create<arith::SelectOp>(loc, mask, vExp, vSafePadPow).getResult();
        
            auto vPowf = rewriter.create<math::PowFOp>(loc, safeBase, safeExp);
            vectorisedMapping[powf.getResult()] = vPowf.getResult();
            rewriter.replaceOp(powf, vPowf.getResult());
            continue;
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
          if (!constOp) return rewriter.notifyMatchFailure(shfl, "Dynamic shuffle offset not supported");
          uint32_t delta = constOp.value();
        
          if (shfl.getMode() == gpu::ShuffleMode::DOWN) {
            SmallVector<int64_t> shuffleMask;
            for (int64_t i = 0; i < vWidth; ++i) shuffleMask.push_back(std::min<int64_t>(i + delta, vWidth - 1));
            auto vShfl = rewriter.create<vector::ShuffleOp>(loc, vInput, vInput, shuffleMask);
            vectorisedMapping[shfl.getResult(0)] = vShfl.getResult();
            rewriter.replaceOp(shfl, vShfl.getResult());
          } else if (shfl.getMode() == gpu::ShuffleMode::XOR) {
            SmallVector<int64_t> xorMask;
            for (int64_t i = 0; i < vWidth; ++i) xorMask.push_back(i ^ delta);
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
            case gpu::AllReduceOperation::MINNUMF: kind = vector::CombiningKind::MINNUMF; break;
            case gpu::AllReduceOperation::MAXNUMF: kind = vector::CombiningKind::MAXNUMF; break;
            case gpu::AllReduceOperation::MINSI: kind = vector::CombiningKind::MINSI; break;
            case gpu::AllReduceOperation::MAXSI: kind = vector::CombiningKind::MAXSI; break;
            case gpu::AllReduceOperation::MINUI: kind = vector::CombiningKind::MINUI; break;
            case gpu::AllReduceOperation::MAXUI: kind = vector::CombiningKind::MAXUI; break;
            case gpu::AllReduceOperation::AND: kind = vector::CombiningKind::AND; break;
            case gpu::AllReduceOperation::OR:  kind = vector::CombiningKind::OR; break;
            case gpu::AllReduceOperation::XOR: kind = vector::CombiningKind::XOR; break;
            default: return rewriter.notifyMatchFailure(sgReduce, "Unsupported reduction kind");
          }
          
          Value scalarReduction = rewriter.create<vector::ReductionOp>(loc, kind, vInput).getResult();
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

          bool isUniformAddress = true;
          for (Value idx : atomic.getIndices()) {
            if (vectorisedMapping.count(idx)) {
              isUniformAddress = false;
              break;
            }
          }

          if (isUniformAddress) {
            Value neutralElement = rewriter.create<arith::ConstantOp>(loc, rewriter.getZeroAttr(vType));
            Value maskedVal = rewriter.create<arith::SelectOp>(loc, mask, vVal, neutralElement).getResult();
            auto reduction = rewriter.create<vector::ReductionOp>(loc, vector::CombiningKind::ADD, maskedVal);
            rewriter.create<memref::AtomicRMWOp>(loc, atomic.getKind(), reduction.getResult(), atomic.getMemref(), atomic.getIndices());

          } else {
            SmallVector<Value, 4> vIndices;
            for (Value idx : atomic.getIndices()) vIndices.push_back(getVecOp(idx));

            for (int64_t i = 0; i < vWidth; ++i) {
              Value laneMask = rewriter.create<vector::ExtractOp>(loc, mask, ArrayRef<int64_t>{i}).getResult();
              
              rewriter.create<scf::IfOp>(loc, laneMask, [&](OpBuilder &b, Location l) {
                Value laneVal = b.create<vector::ExtractOp>(l, vVal, ArrayRef<int64_t>{i}).getResult();
                SmallVector<Value, 4> laneIndices;
                for (Value vIdx : vIndices) {
                  laneIndices.push_back(b.create<vector::ExtractOp>(l, vIdx, ArrayRef<int64_t>{i}).getResult());
                }
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
 
  // =========================================================================
  // Pattern 3: Barrier Fission
  // =========================================================================
  struct BarrierFissionPattern : public OpRewritePattern<scf::ParallelOp> {
    using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(scf::ParallelOp parallelOp, PatternRewriter &rewriter) const override {
      gpu::BarrierOp barrier;
      for (auto &op : *parallelOp.getBody()) {
        if (auto b = dyn_cast<gpu::BarrierOp>(&op)) {
          barrier = b;
          break;
        }
      }
      if (!barrier) return failure(); 

      Location loc = parallelOp.getLoc();
      Value ub = parallelOp.getUpperBound()[0];
      Value iv1 = parallelOp.getInductionVars()[0];

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

      rewriter.setInsertionPoint(parallelOp);
      DenseMap<Value, Value> allocations;
      for (Value val : crossBarrierValues) {
        MemRefType allocType = MemRefType::get({ShapedType::kDynamic}, val.getType());
        Value alloc = rewriter.create<memref::AllocaOp>(loc, allocType, ub);
        allocations[val] = alloc;
      }

      rewriter.setInsertionPointAfter(parallelOp);
      auto secondLoop = rewriter.create<scf::ParallelOp>(loc, parallelOp.getLowerBound(), parallelOp.getUpperBound(), parallelOp.getStep());
      Value iv2 = secondLoop.getInductionVars()[0];

      rewriter.setInsertionPoint(barrier);
      for (Value val : crossBarrierValues) {
        rewriter.create<memref::StoreOp>(loc, val, allocations[val], ValueRange{iv1});
      }

      rewriter.setInsertionPointToStart(secondLoop.getBody());
      DenseMap<Value, Value> loads;
      for (Value val : crossBarrierValues) {
        loads[val] = rewriter.create<memref::LoadOp>(loc, allocations[val], ValueRange{iv2}).getResult();
      }

      Operation *curr = barrier->getNextNode();
      Operation *end = &parallelOp.getBody()->back(); 
      while (curr != end) {
        Operation *next = curr->getNextNode();
        curr->moveBefore(secondLoop.getBody()->getTerminator());
        
        curr->walk([&](Operation *nestedOp) {
          for (OpOperand &operand : nestedOp->getOpOperands()) {
            if (loads.count(operand.get())) {
              operand.set(loads[operand.get()]);
            } else if (operand.get() == iv1) {
              operand.set(iv2); 
            }
          }
        });
        curr = next;
      }

      rewriter.eraseOp(barrier);
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

      int finalWidth = regBitWidth; 

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
