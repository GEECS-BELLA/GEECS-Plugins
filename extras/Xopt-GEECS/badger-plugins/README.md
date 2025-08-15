# Badger-Plugins for GEECS

Environment and Interface plugins for GEECS. These plugins are meant to augment
the default Badger-Plugins folder that is set in the badger config, i.e. the
badger config should not point to this folder. (This is so that the algorithms
and extensions can still be found)

To make badger find these plugins, you need to create a symlinks from the
default Badger-Plugins directory to the `geecs` folders in this repository.
That is:
* symlink from BADGER_PLUGIN_ROOT/environments/geecs to Xopt-GEECS/badger-plugins/environments/geecs
* symlink from BADGER_PLUGIN_ROOT/interfaces/geecs to Xopt-GEECS/badger-plugins/interfaces/geecs

An example of how to create the symlinks on a mac:

ln -s BADGER_PLUGIN_ROOT/environments/geecs to Xopt-GEECS/badger-plugins/environments/geecs
