{ pkgs, lib, ... }:

{
  # ── Runtime: real CUDA driver, not the NixOS stub ──────────────────────
  env.LD_LIBRARY_PATH = "/run/opengl-driver/lib";
  env.AMREX_HOME      = "/home/milk-white-way/frameworks/amrex";
  env.GIO_EXTRA_MODULES = "";

  # ── Build tools ─────────────────────────────────────────────────────────
  packages = with pkgs; [
    gnumake
    openmpi
  ];

  languages.python = {
    enable = true;
    venv.enable = true;
    venv.requirements = ''
      yt
      numpy
      matplotlib
    '';
  };

  # ── Shell scripts ───────────────────────────────────────────────────────
  scripts.ovrflw-build.exec = ''
    cd "$DEVENV_ROOT/Exec"
    make -j$(nproc) "$@"
  '';

  scripts.ovrflw-run.exec = ''
    cd "$DEVENV_ROOT/Exec"
    mpirun -np ''${1:-1} ./ovrflw3d.gnu.TPROF.MPI.CUDA.ex ''${2:-inputs}
  '';

  scripts.ovrflw-clean.exec = ''
    cd "$DEVENV_ROOT/Exec"
    make realclean
  '';

  scripts.ovrflw-test-tgv.exec = ''
    set -e
    cd "$DEVENV_ROOT/Tests/tgv_2d"
    LOG="$DEVENV_ROOT/Tests/tgv_2d/tgv_test_$(date +%Y%m%d_%H%M%S).log"
    echo "  [TEST] Building 2D solver..."
    make -j$(nproc)
    echo "  [TEST] Running 2D Taylor-Green Vortex..."
    mpirun -np 4 ./tgv2d.gnu.TPROF.MPI.ex inputs 2>&1 | tee "$LOG"
    PLT=$(ls -d pltResults00100 2>/dev/null | head -1)
    python "$DEVENV_ROOT/Tests/compare.py" --case tgv --pltfile "$PLT" "$@" 2>&1 | tee -a "$LOG"
    echo "  [TEST] Log saved to $LOG"
  '';

  scripts.ovrflw-test-ldc.exec = ''
    set -e
    cd "$DEVENV_ROOT/Tests/ldc"
    LOG="$DEVENV_ROOT/Tests/ldc/ldc_test_$(date +%Y%m%d_%H%M%S).log"
    echo "  [TEST] Building 2D solver..."
    make -j$(nproc)
    echo "  [TEST] Running 2D Lid-Driven Cavity (Re=400)..."
    mpirun -np 4 ./ldc2d.gnu.TPROF.MPI.ex inputs 2>&1 | tee "$LOG"
    PLT=$(ls -d pltResults20000 2>/dev/null | head -1)
    python "$DEVENV_ROOT/Tests/compare.py" --case ldc --re 400 --pltfile "$PLT" "$@" 2>&1 | tee -a "$LOG"
    echo "  [TEST] Log saved to $LOG"
  '';

  # ── Greeting ────────────────────────────────────────────────────────────
  enterShell = ''
    echo ""
    echo "   ██████╗  ██╗   ██╗ ██████╗  ███████╗ ██╗      ██╗    ██╗"
    echo "  ██╔═══██╗ ██║   ██║ ██╔══██╗ ██╔════╝ ██║      ██║    ██║"
    echo "  ██║   ██║ ██║   ██║ ██████╔╝ █████╗   ██║      ██║ █╗ ██║"
    echo "  ██║   ██║ ╚██╗ ██╔╝ ██╔══██╗ ██╔══╝   ██║      ██║███╗██║"
    echo "  ╚██████╔╝  ╚████╔╝  ██║  ██║ ██║      ███████╗ ╚███╔███╔╝"
    echo "   ╚═════╝    ╚═══╝   ╚═╝  ╚═╝ ╚═╝      ╚══════╝  ╚══╝╚══╝"
    echo ""
    echo "  AMREX_HOME : $AMREX_HOME"
    echo "  MPI        : $(mpirun --version 2>&1 | head -1)"
    echo "  GPU        : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'not detected')"
    echo "  CUDA libs  : $LD_LIBRARY_PATH"
    echo ""
    echo "  Commands   : ovrflw-build  ovrflw-run [nproc] [inputs]  ovrflw-clean"
    echo "  Testing    : ovrflw-test-tgv  ovrflw-test-ldc  [--plot to save PNGs]"
    echo ""
  '';
}
