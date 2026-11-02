{
  description = "Python Developer Environment (uv & ruff)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    devenv.url = "github:cachix/devenv";
    systems.url = "github:nix-systems/default";
  };

  outputs = { self, nixpkgs, devenv, systems, ... }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (system: {
        default = devenv.lib.mkShell {
          inherit inputs;
          pkgs = nixpkgs.legacyPackages.${system};
          modules = [
            {
              devenv.root = let
                pwd = builtins.getEnv "PWD";
              in
                if pwd != "" then pwd else builtins.toString ./.;
            }
            ./devenv.nix
          ];
        };
      });
    };
}
