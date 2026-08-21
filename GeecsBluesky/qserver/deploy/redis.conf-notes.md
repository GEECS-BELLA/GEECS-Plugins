# Redis Notes for the Queueserver Host

The dedicated Ubuntu 22.04 queueserver host should use the distro Redis
package and its systemd unit:

```bash
sudo apt update
sudo apt install redis-server
sudo systemctl enable --now redis-server.service
```

Keep Redis bound to loopback only. The Ubuntu package default is suitable for
the RE Manager control-plane shape:

```conf
bind 127.0.0.1 ::1
protected-mode yes
```

The source-built Redis used during sandbox testing was only a no-sudo
workaround. It is not the deployment shape for the service host.

Redis may warn at startup that `vm.overcommit_memory` is disabled. Apply the
host-level sysctl fix once:

```bash
echo 'vm.overcommit_memory = 1' | sudo tee /etc/sysctl.d/99-redis-overcommit.conf
sudo sysctl --system
```
