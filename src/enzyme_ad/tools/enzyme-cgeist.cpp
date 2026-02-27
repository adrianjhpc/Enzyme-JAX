#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

// Dialect Headers
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

// Polygeist Headers
#include "polygeist/Dialect.h"
#include "polygeist/Passes/Passes.h"
#include "polygeist/Frontend/MIRGenerator.h"
#include "clang/Frontend/CompilerInstance.h"

// Transformation Headers
#include "mlir/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional, llvm::cl::desc("<input file>"), llvm::cl::Required);

int main(int argc, char **argv) {
    llvm::cl::ParseCommandLineOptions(argc, argv, "Enzyme-CGeist GPU Sink Tool\n");

    DialectRegistry registry;
    registry.insert<arith::ArithDialect,
                    affine::AffineDialect,
                    func::FuncDialect,
                    gpu::GPUDialect,
                    memref::MemRefDialect,
                    scf::SCFDialect,
                    LLVM::LLVMDialect,            // CRITICAL: Required for pointers
                    NVVM::NVVMDialect,            // CRITICAL: Required for CUDA
                    polygeist::PolygeistDialect>(); // CRITICAL: Required for raw C++ AST

    MLIRContext context(registry);
    context.loadAllAvailableDialects();

    mlir::polygeist::MIRGenerator generator(context);

    // CRITICAL: Clang needs strict flags to parse CUDA into MLIR
    std::vector<std::string> clangArgs = {
        "-xcuda", 
        "--cuda-device-only",    // Tell Clang we want the kernels, not the host code
        "--cuda-gpu-arch=sm_80",
        "-O3"                    // Polygeist often requires optimizations to resolve C++ templates cleanly
        // "-I/usr/local/cuda/include" // Uncomment if it complains about missing cuda.h
    };

    // Generate the MLIR Module directly from the source filename
    auto module = generator.generate(inputFilename, clangArgs);

    if (!module) {
        llvm::errs() << "Frontend failed to raise source to MLIR. Check Clang CUDA flags.\n";
        return 1;
    }

    // DEBUG: Print the raw Polygeist MLIR so you can see what it looks like before passes!
    // llvm::outs() << "--- RAW POLYGEIST OUTPUT ---\n";
    // module->print(llvm::outs());

    PassManager pm(&context);

    // 1. Polygeist Cleanup Pipeline
    // You MUST lower Polygeist's custom C++ dialect down to standard MLIR
    pm.addPass(polygeist::createCanonicalizeForPass());
    pm.addPass(polygeist::createRaiseSCFToAffinePass());
    
    // Add standard cleanup to resolve pointers to MemRefs
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());

    // 2. Your custom vectorizer/CPU distributor
    // Make sure your pass is actually registered/included properly in your build
    // pm.addPass(createCUDAToHierarchicalParallelPass(256, 4));

    if (failed(pm.run(*module))) {
        llvm::errs() << "Pass pipeline failed\n";
        return 1;
    }
    
    module->print(llvm::outs());
    return 0;
}
