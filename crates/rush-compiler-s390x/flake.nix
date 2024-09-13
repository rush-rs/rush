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
            cargo
            rustc
            clippy
        ];

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
