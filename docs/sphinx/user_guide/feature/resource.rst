.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-resource-label:

=========
Resources
=========

This section describes the basic concepts of resource types and how to use 
them with RAJA-based kernels using ``RAJA::forall``, ``RAJA::kernel``, ``
RAJA::launch``, etc. Resources are used as an interface to various RAJA 
back-end constructs and their respective hardware. Currently there 
exist resource types for ``Cuda``, ``Hip``, ``Omp`` (target) and ``Host``.
Resource objects allow one to allocate and deallocate storage in memory spaces
associated with RAJA back-ends and copy data between memory spaces. They also
allow one to execute RAJA kernels asynchronously on a respective thread/stream.
Resource support in RAJA is rudimentary at this point and its functionality /
behavior may change as it is developed.

.. note:: * Currently feature complete asynchronous behavior and 
            streamed/threaded support is available only for ``Cuda`` and 
            ``Hip`` resources.
          * RAJA resource support is based on camp resource support. The 
            ``RAJA::resources`` namespace aliases the ``camp::resources`` 
            namespace.

Each resource has a set of underlying functionality that is synonymous across 
each resource type.  

 ===================== ===============================================
 Methods               Brief description
 ===================== ===============================================
 get_platform          Returns the underlying camp platform
                       associated with the resource.
 get_event             Return an event object for the resource from
                       the last resource call.
 allocate              Allocate data on a resource back-end.
 deallocate            Deallocate data on a resource back-end.
 memcpy                Perform a memory copy from a source location
                       to a destination location on a resource back-end.
 memset                Set memory value in an allocation on a resource
                       back-end.
 wait                  Wait for all operations enqueued on a resource to
                       complete before proceeding.
 wait_for              Enqueue a wait on a resource stream/thread
                       for a user passed event to complete.
 ===================== ===============================================

.. note:: ``deallocate``, ``memcpy`` and ``memset`` operations only work with 
          pointers that correspond to memory locations that have been 
          allocated on the resource's respective device.
  
Each resource type also defines specific back-end information/functionality. 
For example, each CUDA resource contains a ``cudaStream_t`` value with an 
associated get method. The basic interface for each resource type is 
summarized in `Camp resource <https://github.com/LLNL/camp/blob/main/include/camp/resource.hpp>`_.

.. note:: Stream IDs are assigned to resources in a round robin fashion. The 
          number of independent streams for a given back-end is limited to the 
          maximum number of concurrent streams that the back-end supports. 

------------
Type-Erasure
------------

Resources can be declared in two ways, as a type-erased resource or as a 
concrete resource. The underlying run time functionality is the same for both.

Here is one way to construct a concrete CUDA resource type::

  RAJA::resources::Cuda my_cuda_res;

A type-erased resource allows a user the ability to change the resource 
back-end at run time. For example, to choose a CUDA GPU device resource or 
host resource at run time, one could do the following::

  RAJA::resources::Resource* my_res = nullptr;

  if (use_gpu)
    my_res = new RAJA::resources::Resource{RAJA::resources::Cuda()};
  else
    my_res = new RAJA::resources::Resource{RAJA::resources::Host()};

When ``use_gpu`` is true, ``my_res`` will be a CUDA GPU device resource. 
Otherwise, it will be a host CPU resource.

-------------------
Memory Operations
-------------------

The example discussed in this section illustrates most of the memory
operations that can be performed with RAJA resource objects.
A common use case for a resource is to manage arrays in the appropriate 
memory space to use in a kernel. Consider the following code example::

  // create a resource for a host CPU and a CUDA GPU device
  RAJA::resources::Resource host_res{RAJA::resources::Host()};
  RAJA::resources::Resource cuda_res{RAJA::resources::Cuda()};

  // allocate arrays in host memory and device memory
  int N = 100;

  int* host_array = host_res.allocate<int>(N);
  int* gpu_array = cuda_res.allocate<int>(N);

  // initialize values in host_array....

  // initialize gpu_array values to zero
  cuda_res.memset(gpu_array, 0, sizeof(int) * N);

  // copy host_array values to gpu_array 
  cuda_res.memcpy(gpu_array, host_array, sizeof(int) * N);

  // execute a CUDA kernel that uses gpu_array data
  RAJA::forall<RAJA::cuda_exec<128>>(RAJA::TypedRangeSegment<int>(0, N), 
    [=] RAJA_DEVICE(int i) {
      // modify values of gpu_array...
    }
  );

  // copy gpu_array values to host_array 
  cuda_res.memcpy(host_array, gpu_array, sizeof(int) * N);

  // do something with host_array on CPU...

  // de-allocate array storage
  host_res.deallocate(host_array); 
  cuda_res.deallocate(gpu_array); 

Here, we create a CUDA GPU device resource and a host CPU resource and use
them to allocate an array in GPU memory and one in host memory, respectively.
Then, after initializing the host array, we use the CUDA resource to copy the
host array to the GPU array storage. Next, we run a CUDA device kernel
which modifies the GPU array. After using the CUDA resource to copy the GPU
array values into the host array, we can do something with the values 
generated in the GPU kernel on the CPU host. Lastly, we de-allocate the 
arrays.

--------------------------------
Kernel Execution and Resources
--------------------------------

Resources can be used with the following RAJA kernel execution interfaces:

  * ``RAJA::forall``
  * ``RAJA::kernel``
  * ``RAJA::launch``
  * ``RAJA::sort``
  * ``RAJA::scan``

Although we show examples using mainly ``RAJA::forall`` in the following
discussion, resource usage with the other methods listed is similar and
provides similar behavior.

Usage
^^^^^
 
Specifically, a resource can be passed optionally as the first argument in 
a call to one of these methods. For example::

  RAJA::forall<ExecPol>(my_res, .... );

.. note:: When a resource is not passed when calling one of the methods listed 
          above, the *default* resource type associated with the execution
          policy is used in the internal implementation.

When passing a CUDA or HIP resource, the method will execute asynchronously 
on a GPU stream. Currently, CUDA and HIP are the only resource types that 
enable asynchronous threading. 

.. note:: Support for OpenMP CPU multithreading, which would use the 
          ``RAJA::resources::Host`` resource type, and OpenMP target offload
          which would use the ``RAJA::resources::Omp`` resource type,
          is incomplete and under development.

The resource type passed to one of the methods listed above must be a 
concrete type; i.e., not type erased. The reason is that this allows 
consistency checking via a compile-time assertion to ensure that the passed 
resource is compatible with the given execution policy. For example::
    
  using ExecPol = RAJA::cuda_exec_async<BLOCK_SIZE>;

  RAJA::resources::Cuda my_cuda_res;
  RAJA::forall<ExecPol>(my_cuda_res, .... ); // Successfully compiles

  RAJA::resources::Resource my_res{RAJA::resources::Cuda()};
  RAJA::forall<ExecPol>(my_res, .... )       // Compilation error since resource type is not concrete

  RAJA::resources::Host my_host_res;
  RAJA::forall<ExecPol>(my_host_res, .... )  // Compilation error since resource type is incompatible with the execution policy

IndexSet Usage
^^^^^^^^^^^^^^^
 
Recall that a kernel that uses a RAJA IndexSet to describe the kernel iteration 
space, require a two execution policies (see :ref:`indexsetpolicy-label`). 
Currently, a user may only pass a single resource to a method taking
an IndexSet argument. The resource is used for the *inner* execution over
each segment in the IndexSet, not for the *outer* iteration over segments.
For example::

  using ExecPol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<256>>;
  RAJA::forall<ExecPol>(my_cuda_res, iset,  .... );

Default Resources
^^^^^^^^^^^^^^^^^^^^^^
 
When a resource is not provided by the user, a *default* resource that
corresponds to the execution policy is used. The default resource 
can be accessed in multiple ways. It can be accessed directly from 
the concrete resource type::

  RAJA::resources::Cuda my_default_cuda = RAJA::resources::Cuda::get_default();

The resource type can also be deduced in two different ways from an execution 
policy::

  using Res = RAJA::resources::get_resource<ExecPol>::type;
  Res r = Res::get_default();

Or::

  auto my_resource = RAJA::resources::get_default_resource<ExecPol>();

.. note:: For CUDA and HIP, the default resource is *NOT* associated with the 
          default CUDA or HIP stream. It is its own stream defined by the
          underlying camp resource. This is intentional to break away
          from some issues that arise from the synchronization behavior
          of the CUDA and HIP default streams. It is still possible to use the 
          CUDA and HIP default streams as the default resource. This can be 
          enabled by defining the environment variable 
          ``CAMP_USE_PLATFORM_DEFAULT_STREAM`` before compiling RAJA in a 
          project.

------
Events
------

Event objects allow users to wait or query the status of a resource's action. 
An event can be returned from a resource::

  RAJA::resources::Event e = my_res.get_event();

Getting an event like this enqueues an event object for the given back-end. 

Users can call the *blocking* ``wait`` function on the event::

  e.wait();

This wait call will block all execution until all operations enqueued on a
resource complete.

Alternatively, a user can enqueue the event on a specific resource, forcing 
only the resource to wait for the operation associated with the event to
complete::

  my_res.wait_for(&e);

All methods listed above near the beginning of the RAJA resource discussion
return an event object so users can access the event associated with the 
method call. This allows one to set up dependencies between resource objects 
and operations, as well as define and control asynchronous execution patterns.

.. note:: An Event object is only created if a user explicitly sets the event 
          returned by the ``RAJA::forall`` call to a variable. This avoids 
          unnecessary event objects being created when not needed. For example::
    
            RAJA::forall<cuda_exec_async<BLOCK_SIZE>>(my_cuda_res, ...);

          will *not* generate a cudaStreamEvent, whereas::

            RAJA::resources::Event e = RAJA::forall<cuda_exec_async<BLOCK_SIZE>>(my_cuda_res, ...);

          will generate a cudaStreamEvent.

-------
Example
-------

The example presented here executes three kernels across two CUDA streams on 
a GPU with a requirement that the first and second kernel finish execution 
before the third is launched. It also shows copying memory from the device 
to host on a resource that we described earlier.
    
First, we define two concrete CUDA resources and one concrete host resource,
and define an asynchronous CUDA execution policy type:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_defres_start
   :end-before: _raja_res_defres_end
   :language: C++

Next, we allocate data for two GPU arrays and one host array, all of length 'N':

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_alloc_start
   :end-before: _raja_res_alloc_end
   :language: C++

Then, we launch a GPU kernel on the CUDA stream associated with the resource
``res_gpu1``, without keeping a handle to the associated event:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_k1_start
   :end-before: _raja_res_k1_end
   :language: C++

Next, we execute another GPU kernel on the CUDA stream associated with the
resource ``res_gpu2`` and keep a handle to the corresponding event object 
by assigning it to a local variable ``e``:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_k2_start
   :end-before: _raja_res_k2_end
   :language: C++

We require that the next kernel we launch to wait for the kernel launched on 
the stream associated with the resource ``res_gpu2`` to complete. Therefore, 
we enqueue a wait on that event on the ``res_gpu1`` resource:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_wait_start
   :end-before: _raja_res_wait_end
   :language: C++

Now that the second GPU kernel is complete, we launch a second kernel on the
stream associated with the resource ``res_gpu1``:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_k3_start
   :end-before: _raja_res_k3_end
   :language: C++

Next, we enqueue a memcpy operation on the resource ``res_gpu1`` to copy 
the GPU array ``d_array`` to the host array ``h_array``:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_memcpy_start
   :end-before: _raja_res_memcpy_end
   :language: C++
    
Lastly, we use the copied data in a kernel executed on the host:

.. literalinclude:: ../../../../examples/resource-forall.cpp
   :start-after: _raja_res_k4_start
   :end-before: _raja_res_k4_end
   :language: C++
