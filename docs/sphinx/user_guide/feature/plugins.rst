.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-plugins-label:

========
Plugins
========

RAJA supports user-made plugins that may be loaded either at compilation time
(static plugins) or during runtime (dynamic plugins). These two methods are not 
mutually exclusive, as plugins loaded statically can be run alongside plugins that 
are loaded dynamically.

-------------------
Using RAJA Plugins
-------------------

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Static vs Dynamic Loading
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Static loading** is done at compile time and requires recompilation in order to add, remove, or change a plugin. This is arguably the easier method to implement, requiring only simple file linking to make work. However, recompilation may get tedious and resource-heavy when working with many plugins or on large projects. In these cases, it may be better to load plugins dynamically, requiring no recompilation of the project most of the time.

**Dynamic loading** is done at runtime and only requires the recompilation or 
moving of plugin files in order to add, remove, or change a plugin. This will 
likely require more work to set up, but in the long run may save time and resources.
RAJA checks the environment variable ``RAJA_PLUGINS`` for a path to a plugin or 
plugin directory, and automatically loads them at runtime. This means that a plugin
can be added to a project as easily as making a shared object file and setting 
``RAJA_PLUGINS`` to the appropriate path.

^^^^^^^^^^^^^^^^^^^
Plugins Quick Start
^^^^^^^^^^^^^^^^^^^

**Static Plugins**

#. Build RAJA normally.
#. Use an ``#include`` statement in your code or pass options to the compiler to load your plugin file with your project at compile time. For example: ``g++ project.cpp plugin.cpp -lRAJA -ldl -o project``.
#. When you run your project, your plugin should work.

**Dynamic Plugins**

#. Build RAJA normally.
#. Compile your plugin to be a shared object file with ``.so`` extension. For example: ``g++ plugin.cpp -lRAJA -fPIC -shared -o plugin.so``.
#. Set the environment variable ``RAJA_PLUGINS`` to the path of your ``.so`` file.  This can either be the path to its directory or to the shared object file itself. If the path is a directory, all ``.so`` files in that directory will be loaded.
#. When you run your project, your plugins should work.

^^^^^^^^^^^^^^^^^^^^^^^^^^^
Interfacing with Plugins
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The RAJA plugin API allows for limited interfacing between a project and a 
plugin. There are a couple of methods to call in your code:
``init_plugins`` and ``finalize_plugins``. These will call the corresponding 
``init`` and ``finalize`` methods, respectively, of *every* currently loaded 
plugin. It's worth noting that plugins don't require either an init or finalize
method by default.

* ``RAJA::util::init_plugins();`` will call the ``init`` method of every 
  currently loaded plugin.

* ``RAJA::util::init_plugins("path/to/plugins");`` will call the ``init`` 
  method of every currently loaded plugin and, in addition, will also 
  dynamically load plugins located at the given path.

* ``RAJA::util::finalize_plugins();`` will call the ``finalize`` method of 
  every currently loaded plugin. 


--------------------------
Creating Plugins For RAJA
--------------------------

Plugins are classes derived from the ``RAJA::util::PluginStrategy`` base class
and implement the required virtual methods for the API. An example 
implementation can be found at the bottom of this page.

^^^^^^^^^^^^^^^^^^^^
Plugin API methods
^^^^^^^^^^^^^^^^^^^^

The following list summarizes the virtual methods in the 
``RAJA::util::PluginStrategy`` base class.

* ``void init(const PluginOptions& p) override {}`` is called on all plugins 
  when a user calls ``init_plugins()``

* ``void preCapture(const PluginContext& p) override {}`` is called before 
  lambda capture in RAJA kernel execution methods.

* ``void postCapture(const PluginContext& p) override {}`` is called after 
  lambda capture in RAJA kernel execution methods.

* ``void preLaunch(const PluginContext& p) override {}`` is called before 
  a RAJA kernel execution method runs a kernel.

* ``void postLaunch(const PluginContext& p) override {}`` is called after 
  a RAJA kernel execution method runs a kernel.

* ``void finalize() override {}`` is called on all plugins when a user calls 
  ``finalize_plugins``. This will also unload all currently loaded plugins.

.. note:: The pre/post methods above are automatically called
          before and after executing a kernel with ``RAJA::forall`` or 
          ``RAJA::kernel`` kernel execution methods.

.. note:: The ``init`` and ``finalize`` methods are never called by
          default and are only called when a user calls 
          ``RAJA::util::init_plugins()`` or ``RAJA::util::finalize_plugin()``, 
          respectively.

^^^^^^^^^^^^^^^^^
Static Loading
^^^^^^^^^^^^^^^^^

If a plugin is to be loaded into a project at compile time, it must be
loaded with either an ``#include`` statement in the project source code or
by calling the following method in the project source code, which adds the 
plugin to the RAJA PluginRegistry::

  static RAJA::util::PluginRegistry::add<PluginName> P("Name", "Description");

In either case, the plugin will be loaded every time the compiled 
project executable is run.

^^^^^^^^^^^^^^^^^
Dynamic Loading
^^^^^^^^^^^^^^^^^

If a plugin is to be dynamically loaded in a project at run time, the RAJA 
plugin API requires a few conditions to be met. The following must be true 
about the plugin, not necessarily of the project using it.

#. The plugin must have the following factory method that returns
   a pointer to an instance of your plugin::

     extern "C" RAJA::util::PluginStrategy* RAJAGetPlugin()
     {
       return new MyPluginName;
     }
  
   Note that using ``extern "C"`` is required to search for the ``RAJAGetPlugin()``
   method call for the dynamically loaded plugin correctly.

   Note that the symbol for the factory method has been renamed and the old name
   ``getPlugin`` is deprecated and support for that name will be removed in a future
   release.

#. The plugin must be compiled to be a shared object with a ``.so`` extension. 
   For example: ``g++ plugin.cpp -lRAJA -fPIC -shared -o plugin.so``. 

   At the moment, RAJA will only attempt to load files with ``.so`` extensions. 
   It's worth noting why these flags (or their equivalents) are important. 

     * ``-lRAJA`` is a standard flag for linking the RAJA library. 

     * ``-fPIC`` tells the compiler to produce *position independent code*, 
       which prevents conflicts in the address space of the executable. 

     * ``-shared`` will let the compiler know that you want the resulting 
       object file to be shared, removing the need for a *main* as well as 
       giving dynamically loaded executables access to methods flagged 
       with ``extern "C"``.

#. The ``RAJA_PLUGINS`` environment variable must be set, or the project code 
   must call ``RAJA::util::init_plugins("path");``. Either of these approaches
   is required to supply the path to either a directory containing the plugin
   or its ``.so`` file. It's worth noting that these are not mutually 
   exclusive. RAJA will look for plugins based on the environment variable on 
   program startup and new plugins may be loaded after that by calling the 
   ``init_plugins()`` method.


^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Example Plugin Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following is an example plugin that simply will print out the number of 
times a kernel has been launched and has the ability to be loaded either 
statically or dynamically.

.. literalinclude:: ../../../../examples/plugin/counter-plugin.cpp
   :start-after: _plugin_example_start
   :end-before: _plugin_example_end
   :language: C++

^^^^^^^^^^^^^^^^^^^^^
CHAI Plugin
^^^^^^^^^^^^^^^^^^^^^

RAJA provides abstractions for parallel execution, but does not support
a memory model for managing data in heterogeneous memory spaces. One
option for managing such data is to use `CHAI <https://github.com/LLNL/CHAI>`_,
which provides an array abstraction that integrates with RAJA to enable 
automatic copying of data at runtime to the proper execution memory space for 
a RAJA-based kernel determined by the RAJA execution policy used to execute the 
kernel. Then, the data can be accessed inside the kernel as needed.

To build CHAI with RAJA integration, you need to download and install CHAI with
the ``ENABLE_RAJA_PLUGIN`` option turned on.  Please see 
`CHAI <https://github.com/LLNL/CHAI>`_ for details.

After CHAI has been built with RAJA support enabled, applications can use CHAI
``ManangedArray`` objects to access data inside a RAJA kernel. For example::

  chai::ManagedArray<float> array(1000);

  RAJA::forall<RAJA::cuda_exec<16> >(0, 1000, [=] __device__ (int i) {
      array[i] = i * 2.0f;
  });

  RAJA::forall<RAJA::seq_exec>(0, 1000, [=] (int i) {
    std::cout << "array[" << i << "]  is " << array[i] << std::endl;
  });

Here, the data held by ``array`` is allocated on the host CPU. Then, it is
initialized on a CUDA GPU device. CHAI sees that the data lives on the CPU
and is needed in a GPU device data environment since it is used in a kernel that
will run with a RAJA CUDA execution policy. So it copies the data from
CPU memory to GPU memory, making it available for access in the RAJA kernel. 
The data is printed in the second kernel which runs on the CPU (indicated by the
RAJA sequential execution policy). So CHAI copies the data back to the host CPU.
All necessary data copies are done transparently on demand for each kernel.

