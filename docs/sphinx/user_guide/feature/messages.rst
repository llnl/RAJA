.. ##
.. ## Copyright (c) Lawrence Livermore National Security, LLC and other
.. ## RAJA Project Developers. See top-level LICENSE and COPYRIGHT
.. ## files for dates and other details. No copyright assignment is required
.. ## to contribute to RAJA.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _feat-message-label:

===============================
Messages
===============================

RAJA provides a portable interface and type-safe way to store function arguments that
passed to a function at a later time. For example, from a GPU, arguments can be stored
and passed to function that prints to a file on the CPU. 

.. warning:: This capability is new and should be considered experimental.


--------------------------
How to manage messages?
--------------------------
All messages are handled via the ``message_manager``, which is responsible for
storing callbacks and a list of messages. For the purposes of ``RAJA::messages``, 
a single message can be thought of:

* ``msg_header``: helper data internal to ``RAJA``
* ``msg_args``: a tuple of arguments needed to pass to the function  

To create the ``message_manager``: 

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_manager_start
    :end-before: _raja_msg_manager_end
    :language: C++

``buf_sz`` is the size of the buffer that stores messages. ``host`` is the resource of the 
execution policy, which determines what type of memory to store messages with. For GPU resources,
this is ``PINNED`` memory.

Subscribing callbacks
^^^^^^^^^^^^^^^^^^^^^
To create a specific message type, callbacks must subscribe first. This can be done in two ways:

* ``subscribe(Callable)``: Subscribing with just a callable will create a new type of message with the
  type depending on the parameters.
* ``subscribe(msg_queue_id, Callable)``: Subscribing with ``msg_queue_id`` and a callable will append the new 
  callback to the already existing callback list.

As an example for subscribing with both methods:

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_subscribe_start
    :end-before: _raja_msg_subscribe_end
    :language: C++

Unsubscribing callbacks
^^^^^^^^^^^^^^^^^^^^^^^
If a particular callback no longer needs to be subscribed to a message type, then the callback can be 
unsubcribed. This can be achieved in three ways:

* ``unsubscribe(msg_queue_id, Callable)``: Looks for a specific callback that is subscribed to a particular message. If 
  the callback is subscribed, remove from callback list. Otherwise, throws an exception.  
* ``unsubscribe_all(msg_queue_id)``: Removes all callbacks subscribed to a particular message
* ``unsubscribe_all()``: Removes all callbacks and messages.

An example for unsubscribing a callback:

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_unsubscribe_start
    :end-before: _raja_msg_unsubscribe_end
    :language: C++


Publishing messages
^^^^^^^^^^^^^^^^^^^
Messages can be published/stored in a ``msg_queue``. These are non-owning adapters to the ``msg_bus``, which is 
responsible for storing all messages. The ``msg_queue`` will contain additional type information as well as 
the ``msg_queue_id``. A queue is created once a callback is subscribed to a new message type. Since ``msg_queue`` is 
non-owning, these can be copied. 

Here is how the ``msg_queue`` can be used to publish messages:

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_k2_start
    :end-before: _raja_msg_k2_end
    :language: C++

Handling messages
^^^^^^^^^^^^^^^^^
Lastly, there needs to be a way to "handle" messages. In others, direct the messages to the corresponding callback(s).
This is handled with the ``message_manager``, which forces a synchronize on the resource provided.

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_wait_start
    :end-before: _raja_msg_wait_end
    :language: C++

Handling messages with a GPU
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a complete example using the ``RAJA::messages`` with a GPU kernel.

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_gpu1_start
    :end-before: _raja_msg_gpu1_end
    :language: C++

.. note::
  In this example, ``gpu_policy`` depends on the build (i.e., CUDA, HIP), ``res`` is the default resource for
  ``gpu_policy``, and ``d_*`` arrays are allocated for the device. These are removed above from the example to just 
  show the ``RAJA::messages`` interface.

Handling messages across multiple streams
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Here is a complete example using the ``RAJA::messages`` with multiple resources.

.. literalinclude:: ../../../../examples/messages-forall.cpp
    :start-after: _raja_msg_gpu2_start
    :end-before: _raja_msg_gpu2_end
    :language: C++

.. note::
  In this example, ``res_gpu1`` and ``res_gpu2`` depend on the build (i.e., CUDA, HIP), ``EXEC_POLICY`` the
  exeuction policy for the loops (also depends on the build), and ``d_*`` arrays are allocated for the device
  while ``h_*`` are allocated for the host. These are removed from the example above to just show the 
  ``RAJA::messages`` interface.

Message queue policies
^^^^^^^^^^^^^^^^^^^^^^
Message queues can support various policies depending on the requirements of that queue, such as
the number of producers/consumers or the type of atomic operations.

 ======================= ============================
 Message queue Policies  Brief description
 ======================= ============================
 spsc                    Supports a single producer,
                         single consumer; i.e., no
                         atomic operations 
 mpsc                    Supports multiple producers,
                         a single consumer; i.e., 
                         requires atomic operations. 
                         Automiatically determines 
                         which atomic operations to 
                         use.
 ======================= ============================

.. note:: Producers and consumers can not operate at the same time

Building and running the example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The example ``examples/messages-forall.cpp`` is built from the RAJA source tree when:

* ``ENABLE_EXAMPLES=On``

To run the example:

.. code-block:: bash

  ./bin/messages-forall
  
This example will show how callbacks can be subscribed to various types of messages as
well as how to publish messages on multiple different platforms. For the purposes of 
this example, output for messages will be printed using ``std::cout``.

--------------------------
Application considerations
--------------------------

Things to consider when using ``RAJA::messages`` in an application.

* The ``msg_queue`` with the correct argument types is created when a callback subscribes. Certain
  patterns will cause this storage to slowly grow overtime. For example, creating a new
  ``msg_queue`` every function call within a loop. Therefore, applications that use this pattern
  will want to unsubscribe at some point to avoid running out of memory. 
* Upon creation of the ``message_manager``, the ``message_bus`` will be allocated with some size. This
  can be resized; however, resizing will force a synchronize and will loss any messages currently stored.
  Also, the allocation is done through the resource, which can be less performant depending on the resource. 
* Since the ``msg_queue`` is a fixed size, there is a chance of lossing messages. The ``try_post_message`` function
  will return a ``boolean``. This will be ``true`` if the message is successfully added to the queue; otherwise, this
  is ``false``. 
