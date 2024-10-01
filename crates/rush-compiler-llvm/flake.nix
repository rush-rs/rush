{
  description = "Environment for developing and deploying the rush LLVM project";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
      };
    in {
      devShells.default = pkgs.mkShell {
        name = "Rush Dev";

        buildInputs = with pkgs; [
          # Misc
            ripgrep
            llvmPackages_14.llvm
            libxml2
            libffi
            cargo
            rustc
        ];

        # LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [pkgs.stdenv.cc.cc pkgs.cudaPackages.cudatoolkit pkgs.cudaPackages.cudnn];
        LLVM_SYS_140_PREFIX = "${pkgs.llvmPackages_14.llvm.dev}";

        shellHook = ''
          # if running from zsh, reenter zsh
          if [[ $(ps -e | grep $PPID) == *"zsh" ]]; then
            export SHELL=zsh
            zsh
            exit
          fi
        '';
      };

      formatter = nixpkgs.legacyPackages.${system}.alejandra;
    });
}
