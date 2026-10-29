# NixOS module: nightly retrain + refresh the Jellyfin collection.
#
# Add to your configuration:
#   imports = [ /root/hvad-skal-vi-seerr/nix/service.nix ];
# then `nixos-rebuild switch`.
#
# Credentials are read from .env.local in the repo (KEY=VALUE per line, no
# `export`) - systemd's EnvironmentFile wants exactly that format, which is
# why no `set -a` dance is needed here.
{ pkgs, ... }:

let
  repo = "/root/hvad-skal-vi-seerr";

  # A pinned interpreter rather than `nix-shell -p` at runtime: the timer then
  # never re-evaluates nixpkgs, needs no network, and can't drift between runs.
  python = pkgs.python3.withPackages (ps: with ps; [ numpy pandas requests ]);

  stage = name: "${python}/bin/python3 -m pipeline.${name}";
in
{
  systemd.services.hvad-skal-vi-seerr = {
    description = "Retrain the Jellyfin recommender and refresh its collection";

    # Both run on this host, so don't bother starting before they're up.
    after = [ "network-online.target" "jellyfin.service" "seerr.service" ];
    wants = [ "network-online.target" ];

    serviceConfig = {
      Type = "oneshot";
      WorkingDirectory = repo;
      EnvironmentFile = "${repo}/.env.local";

      # Type=oneshot runs these in order and skips the rest if one fails, so a
      # failed validation stops the retrain - same intent as the && chain.
      ExecStart = [
        (stage "export_history")
        (stage "validate_library")
        (stage "train_local")
        (stage "recommend")
      ];
    };
  };

  systemd.timers.hvad-skal-vi-seerr = {
    description = "Nightly run of the Jellyfin recommender";
    wantedBy = [ "timers.target" ];

    timerConfig = {
      OnCalendar = "04:00";
      # Catch up after downtime instead of silently skipping a night.
      Persistent = true;
      RandomizedDelaySec = "15m";
    };
  };
}
