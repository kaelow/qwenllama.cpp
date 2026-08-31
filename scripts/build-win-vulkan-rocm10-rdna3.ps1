param(
    [string]$OutputDir = "release-packages",
    [string]$PackageName = "build-win-vulkan-rocm10-rdna3",
    [string]$BuildName = "build-win-vulkan-rocm10-rdna3",
    [string]$Target = "",
    [string]$RocmRoot = "",
    [int]$Parallel = 16,
    [switch]$Package = $false,
    [switch]$AllTests = $false,
    [switch]$SkipStage = $false,
    [switch]$ConfigureOnly = $false,
    [switch]$SkipDeviceCheck = $false
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
$env:MSBUILDDISABLENODEREUSE = "1"

$repoRoot = $PSScriptRoot | Split-Path -Parent
$vsInstaller = "C:\Program Files (x86)\Microsoft Visual Studio\Installer"
$vswhereExe = Join-Path $vsInstaller "vswhere.exe"
$vsInstallPath = $null
if (Test-Path -LiteralPath $vswhereExe) {
    # ROCm 10's bundled Clang 23 is validated with the VS 2022 (v17) STL.
    # VS 2026/MSVC 14.51 declares several cmath builtins as host+device and
    # currently collides with Clang's HIP wrappers.  Prefer v17 when both are
    # installed, then fall back to the newest usable C++ installation.
    $vsInstallPath = & $vswhereExe -latest -version '[17.0,18.0)' -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath | Select-Object -First 1
    if (-not $vsInstallPath) {
        $vsInstallPath = & $vswhereExe -latest -products * `
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
            -property installationPath | Select-Object -First 1
    }
}

$vcvarsPath = if ($vsInstallPath) {
    Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
} else {
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"
}
if (-not (Test-Path -LiteralPath $vcvarsPath)) {
    throw "MSVC x64 build environment was not found (needed for Windows SDK headers and rc.exe)"
}
Write-Host "[ENV] Activating MSVC/Windows SDK via vcvarsall.bat x64"
cmd /c "`"$vcvarsPath`" x64 > nul && set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
}

$ninjaExe = Get-Command ninja.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty Source
if (-not $ninjaExe -and $vsInstallPath) {
    $ninjaExe = Join-Path $vsInstallPath "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
}
if (-not $ninjaExe -or -not (Test-Path -LiteralPath $ninjaExe)) {
    throw "ninja.exe was not found"
}

$rcExe = Get-Command rc.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty Source
if (-not $rcExe -or -not (Test-Path -LiteralPath $rcExe)) {
    throw "Windows SDK rc.exe was not found after activating vcvarsall"
}

$vulkanSdk = $env:VULKAN_SDK
if (-not $vulkanSdk -or -not (Test-Path -LiteralPath $vulkanSdk)) {
    $installedSdks = @(
        Get-ChildItem "C:\VulkanSDK" -Directory -ErrorAction SilentlyContinue |
            Sort-Object { [version]$_.Name } -Descending
    )
    if ($installedSdks.Count -gt 0) {
        $vulkanSdk = $installedSdks[0].FullName
    }
}
if (-not $vulkanSdk -or -not (Test-Path -LiteralPath (Join-Path $vulkanSdk "Bin\glslc.exe"))) {
    throw "Vulkan SDK/glslc.exe was not found; set VULKAN_SDK or install it under C:\VulkanSDK"
}

if (-not $RocmRoot) {
    $rocmCandidates = @()
    if ($env:ROCM_PATH) {
        $rocmCandidates += $env:ROCM_PATH
    }
    $rocmCandidates += "C:\TheRock\build"
    $legacyRoot = "C:\Program Files\AMD\ROCm"
    if (Test-Path -LiteralPath $legacyRoot) {
        $rocmCandidates += @(
            Get-ChildItem -LiteralPath $legacyRoot -Directory -ErrorAction SilentlyContinue |
                Sort-Object Name -Descending |
                Select-Object -ExpandProperty FullName
        )
    }
    $RocmRoot = $rocmCandidates |
        Where-Object { $_ -and (Test-Path -LiteralPath $_) } |
        Select-Object -First 1
}
if (-not $RocmRoot -or -not (Test-Path -LiteralPath $RocmRoot)) {
    throw "ROCm was not found. Pass -RocmRoot for a TheRock build or legacy ROCm installation"
}
$RocmRoot = (Resolve-Path -LiteralPath $RocmRoot).Path

$clangExe = @(
    (Join-Path $RocmRoot "lib\llvm\bin\clang.exe"),
    (Join-Path $RocmRoot "llvm\bin\clang.exe"),
    (Join-Path $RocmRoot "bin\clang.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$clangxxExe = @(
    (Join-Path $RocmRoot "lib\llvm\bin\clang++.exe"),
    (Join-Path $RocmRoot "llvm\bin\clang++.exe"),
    (Join-Path $RocmRoot "bin\clang++.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$deviceLibDir = @(
    (Join-Path $RocmRoot "lib\llvm\amdgcn\bitcode"),
    (Join-Path $RocmRoot "amdgcn\bitcode"),
    (Join-Path $RocmRoot "lib\amdgcn\bitcode")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
$hipccExe = @(
    (Join-Path $RocmRoot "bin\hipcc.bat"),
    (Join-Path $RocmRoot "bin\hipcc.exe")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $clangExe -or -not $clangxxExe -or -not $deviceLibDir -or -not $hipccExe) {
    throw "ROCm at $RocmRoot is missing Clang, HIP, or AMDGPU device bitcode"
}

$rocmBin = Join-Path $RocmRoot "bin"
$rocmLib = Join-Path $RocmRoot "lib"
$env:ROCM_PATH = $RocmRoot
$env:HIP_PATH = $RocmRoot
$env:VULKAN_SDK = $vulkanSdk
$env:PATH = "$(Split-Path -Parent $ninjaExe);$vulkanSdk\Bin;$rocmBin;$(Split-Path -Parent $clangxxExe);$env:PATH"

$hipInfoExe = Join-Path $rocmBin "hipInfo.exe"
if (-not $SkipDeviceCheck) {
    if (-not (Test-Path -LiteralPath $hipInfoExe)) {
        throw "hipInfo.exe is missing; pass -SkipDeviceCheck only for a cross-build host"
    }
    $hipInfo = (& $hipInfoExe 2>&1 | Out-String)
    $gfx1100Ordinal = $null
    $reportedOrdinal = $null
    foreach ($line in ($hipInfo -split "`r?`n")) {
        if ($line -match "^device#\s+(\d+)") {
            $reportedOrdinal = [int]$matches[1]
        } elseif ($line -match "^gcnArchName:\s+gfx1100" -and $null -ne $reportedOrdinal) {
            $gfx1100Ordinal = $reportedOrdinal
            break
        }
    }
    if ($LASTEXITCODE -ne 0 -or $null -eq $gfx1100Ordinal) {
        throw "No gfx1100 device was reported by ROCm. This package is deliberately limited to validated RDNA3 gfx1100"
    }

    # On current native-Windows ROCm stacks a visible unsupported iGPU can
    # make code-object loading fail even after hipSetDevice selects gfx1100.
    # Restrict the build/test process before its first in-process HIP call.
    # Preserve an explicit user filter that already exposes gfx1100.
    if (-not $env:HIP_VISIBLE_DEVICES) {
        $env:HIP_VISIBLE_DEVICES = "$gfx1100Ordinal"
        Write-Host "[ENV] HIP_VISIBLE_DEVICES=$gfx1100Ordinal isolates gfx1100 from other adapters"
    } else {
        Write-Host "[ENV] Preserving HIP_VISIBLE_DEVICES=$($env:HIP_VISIBLE_DEVICES) (gfx1100 visible)"
    }

    # Enumeration alone is insufficient on native Windows: a mismatched
    # Adrenalin/OEM display driver can report gfx1100 normally and then reject
    # every SDK-produced code object with hipErrorInvalidImage.  Compile and
    # launch one real kernel before spending time on the full backend build.
    $smokeDir = Join-Path $repoRoot $BuildName
    $smokeSource = Join-Path $repoRoot "scripts\rocm\hip-runtime-smoke.cpp"
    $smokeExe = Join-Path $smokeDir "hip-runtime-smoke.exe"
    New-Item -ItemType Directory -Path $smokeDir -Force | Out-Null
    Write-Host "[CHECK] Compiling and launching gfx1100 HIP and hipBLAS kernels"
    & $hipccExe "--offload-arch=gfx1100" `
        "--rocm-path=$($RocmRoot -replace '\\','/')" `
        "--rocm-device-lib-path=$($deviceLibDir -replace '\\','/')" `
        $smokeSource -O2 "-L$($rocmLib -replace '\\','/')" -lhipblas -o $smokeExe
    if ($LASTEXITCODE -ne 0) {
        throw "The ROCm compiler could not build the gfx1100 runtime smoke test"
    }
    & $smokeExe
    if ($LASTEXITCODE -ne 0) {
        throw @"
ROCm enumerated the RX 7900 XTX but could not launch a gfx1100 kernel.
The ROCm 10 release notes require a compatible compute driver (Adrenalin
26.6.4 or Windows OEM 26.10.28). Verify the driver, reboot, and rerun this
script; -SkipDeviceCheck is only for cross-building.
"@
    }
}

if ($AllTests) {
    $SkipStage = $true
}

$buildDir = Join-Path $repoRoot $BuildName
$pkgDir = Join-Path $repoRoot "$OutputDir\$PackageName"
$binDir = Join-Path $buildDir "bin"
$targets = @(
    $Target -split '[,;]' |
        ForEach-Object { $_.Trim() } |
        Where-Object { $_ }
)

$cmakeClang = $clangExe -replace '\\', '/'
$cmakeClangxx = $clangxxExe -replace '\\', '/'
$cmakeRc = $rcExe -replace '\\+', '/'
$cmakeRocmRoot = $RocmRoot -replace '\\', '/'
$cmakeNinja = $ninjaExe -replace '\\', '/'
$rocmFlags = "--rocm-path=$($RocmRoot -replace '\\','/') --rocm-device-lib-path=$($deviceLibDir -replace '\\','/')"
$commonFlags = @(
    "-G", "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_C_COMPILER=$cmakeClang",
    "-DCMAKE_CXX_COMPILER=$cmakeClangxx",
    "-DCMAKE_RC_COMPILER=$cmakeRc",
    "-DCMAKE_CXX_FLAGS=$rocmFlags",
    "-DCMAKE_PREFIX_PATH=$cmakeRocmRoot",
    "-DROCM_PATH=$cmakeRocmRoot",
    "-DGPU_TARGETS=gfx1100",
    "-DGGML_VULKAN=ON",
    "-DGGML_HIP=ON",
    "-DGGML_HIP_SPEC_ACCEL=ON",
    "-DGGML_CUDA_FA=ON",
    "-DGGML_CUDA_KVARN=ON",
    "-DGGML_NATIVE=OFF",
    "-DGGML_BACKEND_DL=ON",
    "-DGGML_RPC=ON",
    "-DLLAMA_BUILD_BORINGSSL=ON",
    "-DLLAMA_BUILD_EXAMPLES=ON",
    "-DLLAMA_BUILD_TESTS=ON",
    "-DLLAMA_BUILD_SERVER=ON",
    "-DLLAMA_BUILD_TOOLS=ON"
)

$buildsServer = $targets.Count -eq 0 -or $targets -contains "llama-server"
if ($buildsServer) {
    # Fork build numbers do not identify a compatible upstream UI bucket.
    $commonFlags += "-DLLAMA_BUILD_UI=ON"
    $commonFlags += "-DLLAMA_USE_PREBUILT_UI=OFF"
}

Write-Host "========================================"
Write-Host "BeeLlama Windows Vulkan + ROCm RDNA3 Build"
Write-Host "Vulkan: $vulkanSdk"
Write-Host "ROCm:   $RocmRoot"
Write-Host "Clang:  $clangxxExe"
Write-Host "HIP:    $hipccExe"
Write-Host "Target: gfx1100 (RX 7900 XTX)"
Write-Host "Build:  $buildDir"
Write-Host "Stage:  $(if ($Package) { $pkgDir } else { 'disabled (use -Package)' })"
Write-Host "Jobs:   $Parallel"
Write-Host "========================================"

$cmakeArgs = @("-S", $repoRoot, "-B", $buildDir)
$cmakeArgs += $commonFlags
$cmakeArgs += "-DCMAKE_MAKE_PROGRAM=$cmakeNinja"
Write-Host "`n[CONFIGURE] cmake -S $repoRoot -B $buildDir"
$previousEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @cmakeArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $previousEap
if ($LASTEXITCODE -ne 0) {
    throw "CMake configure failed with exit code $LASTEXITCODE"
}
if ($ConfigureOnly) {
    Write-Host "[DONE] ConfigureOnly requested"
    exit 0
}

$buildArgs = @("--build", $buildDir, "--parallel", "$Parallel")
if ($targets.Count -gt 0) {
    $buildArgs += @("--target") + $targets
} elseif ($AllTests) {
    $ninjaTargets = & $ninjaExe -C $buildDir -t targets all 2>$null
    $testTargets = $ninjaTargets |
        ForEach-Object { if ($_ -match "^(test-[A-Za-z0-9_.-]+):") { $matches[1] } } |
        Where-Object { $_ -notlike "*.exe" } |
        Sort-Object -Unique
    if ($testTargets) {
        $buildArgs += @("--target") + $testTargets
    }
    $buildArgs += @("--", "-k", "0")
}
Write-Host "`n[BUILD] cmake $($buildArgs -join ' ')"
$previousEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& cmake @buildArgs 2>&1 | ForEach-Object { Write-Host $_ }
$ErrorActionPreference = $previousEap
if ($LASTEXITCODE -ne 0) {
    throw "Build failed with exit code $LASTEXITCODE"
}
Write-Host "[OK] Vulkan and HIP backends built into one dynamic-backend package"

if (-not $Package -or $SkipStage) {
    Write-Host "[DONE] Package staging skipped"
    exit 0
}

Write-Host "`n[STAGE] Copying binaries and ROCm gfx1100 runtime to $pkgDir"
New-Item -ItemType Directory -Path $pkgDir -Force | Out-Null
$stageCount = 0
foreach ($pattern in @(
    "$binDir\ggml*.dll",
    "$binDir\llama.dll",
    "$binDir\llama-common.dll",
    "$binDir\*-impl.dll",
    "$binDir\mtmd.dll",
    "$binDir\*.exe",
    "$binDir\libomp*.dll"
)) {
    foreach ($file in Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue) {
        Copy-Item -LiteralPath $file.FullName -Destination $pkgDir -Force
        $stageCount++
    }
}

# TheRock 10 uses versioned HIP/RTC DLLs and a gfx-specific rocBLAS kpack.
# Legacy layouts use the same logical components with occasionally different
# version suffixes, so discover only the required families instead of copying
# the entire ROCm distribution.
foreach ($pattern in @(
    "amdhip64*.dll", "amd_comgr.dll", "hipblas.dll", "rocblas.dll", "rocsolver.dll",
    "rocm_kpack.dll", "libhipblaslt.dll", "origami.dll", "hiprtc*.dll"
)) {
    foreach ($file in Get-ChildItem -LiteralPath $rocmBin -Filter $pattern -ErrorAction SilentlyContinue) {
        Copy-Item -LiteralPath $file.FullName -Destination $pkgDir -Force
        $stageCount++
    }
}
$kpackSource = Join-Path $RocmRoot ".kpack\blas_lib_gfx1100.kpack"
$hasRocblasPayload = $false
if (Test-Path -LiteralPath $kpackSource) {
    $kpackDestination = Join-Path $pkgDir ".kpack"
    New-Item -ItemType Directory -Path $kpackDestination -Force | Out-Null
    Copy-Item -LiteralPath $kpackSource -Destination $kpackDestination -Force
    $stageCount++
    $hasRocblasPayload = $true
}

# rocBLAS on TheRock resolves Tensile assets relative to rocblas.dll under
# bin\rocblas\library. Legacy packages commonly place the same payload under
# lib\rocblas\library. Stage the validated gfx1100 slice in the exact runtime
# layout even when a .kpack is also present: current Windows rocBLAS does not
# automatically substitute that archive for its neighboring lazy library.
$rocblasLibrarySource = @(
    (Join-Path $rocmBin "rocblas\library"),
    (Join-Path $RocmRoot "lib\rocblas\library")
) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if ($rocblasLibrarySource) {
    $rocblasDestination = Join-Path $pkgDir "rocblas\library"
    New-Item -ItemType Directory -Path $rocblasDestination -Force | Out-Null
    $rocblasFiles = @(Get-ChildItem -LiteralPath $rocblasLibrarySource -File |
        Where-Object {
            $_.Name -match 'gfx1100' -or
            $_.Name -in @('TensileLibrary.dat', 'TensileLibrary_lazy.dat')
        })
    foreach ($file in $rocblasFiles) {
        Copy-Item -LiteralPath $file.FullName -Destination $rocblasDestination -Force
        $stageCount++
    }
    if ($rocblasFiles.Count -gt 0) {
        $hasRocblasPayload = $true
    }
}
if (-not $hasRocblasPayload) {
    Write-Warning "No gfx1100 rocBLAS kernel payload was found; the staged HIP backend may require the installed ROCm tree"
}

if (-not $SkipDeviceCheck -and (Test-Path -LiteralPath $smokeExe)) {
    $packagedSmoke = Join-Path $pkgDir "hip-runtime-smoke.exe"
    Copy-Item -LiteralPath $smokeExe -Destination $packagedSmoke -Force
    $stageCount++

    # Validate with no ROCm installation directories on PATH so a missing DLL
    # or Tensile payload cannot be masked by the developer environment.
    Write-Host "[CHECK] Launching packaged HIP + hipBLAS smoke test"
    $savedPath = $env:PATH
    $savedLocation = Get-Location
    try {
        $env:PATH = "$pkgDir;$($env:SystemRoot)\System32;$($env:SystemRoot)"
        Set-Location -LiteralPath $pkgDir
        & $packagedSmoke
        if ($LASTEXITCODE -ne 0) {
            throw "The staged package failed its isolated HIP + hipBLAS runtime smoke test"
        }
    } finally {
        Set-Location -LiteralPath $savedLocation
        $env:PATH = $savedPath
    }
}

foreach ($license in @("LICENSE", "licenses\BridgeSpec-MIT.txt")) {
    $source = Join-Path $repoRoot $license
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination $pkgDir -Force
        $stageCount++
    }
}
Write-Host "[OK] Staged $stageCount files"
Write-Host "[DONE] Package folder: $pkgDir"
