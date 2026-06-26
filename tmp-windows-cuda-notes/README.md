\# RAJA Build with CUDA on Windows

\## MS toolchain

\### Install Visual Studio

2026   https://visualstudio.microsoft.com/downloads/
2022   https://aka.ms/vs/17/release/vs\_community.exe

Steve : tried both initially to see if one would show fewer errors.   The errors seem to be mostly the same for both so used 2026.

NOTE: MS seems to remove references to old VS community versions, found above 2022 link on reddit not sure how long links remain for the old versions.
Paid subscribers can get access to the old VS versions.

\### Install Cmake

https://cmake.org/download/

\### Install Git

https://git-scm.com/install/windows

\### Install CUDA

https://developer.nvidia.com/cuda-downloads

Steve: Tried both CUDA 12 and CUDA 13.   Both had issues so went with CUDA 13 since we have added support for it.

\##  Git stuff

All generic stuff.   Added a key for GitHub.

```
ssh-keygen -t ed25519 -C "smith84@llnl.gov"
type %userprofile%\\.ssh\\id\_ed25519.pub | clip

git clone --recursive git@github.com:llnl/RAJA.git
```

\## Failed attempts to build.

Steve : Tried combinations of MSVCC 2022 and 2026 with Cuda 12 and 13 to see if some combination might work....needed to start deeper patching.

\## Steve patching

\### BLT

Apply the blt.patch to fix argument passing to MSVCC.   nvcc does not pass "/arg" to MSVCC by default which causes nvcc to interpret the arguments as directories/files.

Kenny has similar patch for blt in AXOM branch.

```

cd blt

git apply ..\\tmp-windows-cuda-notes\\blt.patch

```

\### Camp 

MSVCC has issue parsing variadic macros.   Uses non-standard CPP parser.   Did come up with patch for CAMP but Nvidia CUDA 13 recommends using the "-Xcompiler=/Zc:preprocessor" flag which uses more standard conforming CPP.   Seemed more reasonable approach.

Camp patch is in camp-variadic.patch.

\## RAJA config/build

CUDA 13 recommends the preprocessor flag and fix a CAMP issue.   MSVCC uses some non-standard conforming parsing if this is not done.

"/TP" treats .C files as C++.   Was fixing some issues with CUDA 12 toolkit during initial attempts but should double check this requirement.

```
cmake -B cudabuild RAJA -DENABLE\_CUDA=ON -DCMAKE\_CUDA\_FLAGS=" --keep  -Xcompiler=/TP -Xcompiler=/Zc:preprocessor" -DCMAKE\_CUDA\_ARCHITECTURES=75

```

```
cmake --build cudabuild --target atomic-histogram\_solution > zz.build.txt 2>\&1
```

Steve: reached a point with a template issue, which above build exposes.

```
&#x20;atomic-histogram\_solution.cpp
&#x20; atomic-histogram\_solution.cudafe1.cpp
C:\\Users\\smith84\\projects\\raja\\clean\\RAJA\\include\\RAJA/index/IndexSet.hpp(754): warning C4244: 'return': conversion from '\_\_int64' to 'int', possible
&#x20;loss of data \[C:\\Users\\smith84\\projects\\raja\\clean\\cudabuild\\exercises\\atomic-histogram\_solution.vcxproj]
C:\\Users\\smith84\\projects\\raja\\clean\\RAJA\\include\\RAJA/index/IndexSet.hpp(759): warning C4244: 'return': conversion from 'const \_\_int64' to 'int', po
ssible loss of data \[C:\\Users\\smith84\\projects\\raja\\clean\\cudabuild\\exercises\\atomic-histogram\_solution.vcxproj]
C:\\Users\\smith84\\projects\\raja\\clean\\cudabuild\\exercises\\atomic-h.9E0645A7\\x64\\Debug\\atomic-histogram\_solution.cudafe1.stub.c(37): error C2912: expli
cit specialization 'void RAJA::policy::cuda::impl::\_\_wrapper\_\_device\_stub\_forallp\_cuda\_kernel<RAJA::policy::cuda::cuda\_exec\_explicit<RAJA::iteration\_
mapping::Direct,RAJA::cuda::IndexGlobal<RAJA::named\_dim::x,256,0>,RAJA::CudaDefaultConcretizer,1,false>,1,RAJA::Iterators::numeric\_iterator<StorageT,
DiffT,Type \*>,\_\_nv\_dl\_wrapper\_t<\_\_nv\_dl\_tag<int (\_\_cdecl \*)(int,char \*\*),\&int main(int,char \*\*),1>,int \*,int \*>,int,RAJA::expt::ForallParamPack<>,RAJ
A::iteration\_mapping::Direct,RAJA::cuda::IndexGlobal<RAJA::named\_dim::x,256,0>,256>(const \_ZN4camp5decayIZ4mainEUliE0\_EE \&,const \_ZN4camp5decayIN4RAJ
A9Iterators16numeric\_iteratorIiiPiEEEE \&,const \_ZN4camp5decayIiEE \&,\_ZN4camp5decayIN4RAJA4expt15ForallParamPackIJEEEEE \&)' is not a specialization of
&#x20;a function template \[C:\\Users\\smith84\\projects\\raja\\clean\\cudabuild\\exercises\\atomic-histogram\_solution.vcxproj]
&#x20;         with
&#x20;         \[
&#x20;             StorageT=int,
&#x20;             DiffT=int,
&#x20;             Type=int
&#x20;         ]
C:\\Program Files\\Microsoft Visual Studio\\18\\Community\\MSBuild\\Microsoft\\VC\\v180\\BuildCustomizations\\CUDA 13.3.targets(804,9): error MSB3721: The comm
and ""C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\\bin\\nvcc.exe"  --use-local-env -ccbin "C:\\Program Files\\Microsoft Visual Studio\\18\\Com
munity\\VC\\Tools\\MSVC\\14.51.36231\\bin\\HostX64\\x64" -x cu   -IC:\\Users\\smith84\\projects\\raja\\clean\\RAJA\\include -IC:\\Users\\smith84\\projects\\raja\\clean\\
cudabuild\\include -IC:\\Users\\smith84\\projects\\raja\\clean\\RAJA\\tpl\\camp\\include -IC:\\Users\\smith84\\projects\\raja\\clean\\cudabuild\\tpl\\camp\\include -I"C
:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\\include" -I"C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\\include\\cccl" -I"C:\\Prog
ram Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v13.3\\include"  -G  --keep --keep-dir atomic-h.9E0645A7\\x64\\Debug  -maxrregcount=0     --machine 64 --com
pile -cudart static -forward-unknown-to-host-compiler -restrict --expt-extended-lambda --expt-relaxed-constexpr -Xcudafe --display\_error\_number -O0 -
std=c++20 --generate-code=arch=compute\_75,code=\[compute\_75,sm\_75] -Xcompiler="/TP /Zc:preprocessor" -g  -DRAJA\_WIN\_STATIC\_BUILD -DCAMP\_WIN\_STATIC\_BUI
LD -D"CMAKE\_INTDIR=\\"Debug\\"" -D\_MBCS -DWIN32 -D\_WINDOWS -DRAJA\_WIN\_STATIC\_BUILD -DCAMP\_WIN\_STATIC\_BUILD -D"CMAKE\_INTDIR=\\"Debug\\"" -Xcompiler "/EHsc
&#x20;/W3 /nologo  /FS /Zi /RTC1 /MTd /GR" -Xcompiler "/Fdatomic-histogram\_solution.dir\\Debug\\vc145.pdb" -o atomic-histogram\_solution.dir\\Debug\\atomic-his
togram\_solution.obj "C:\\Users\\smith84\\projects\\raja\\clean\\RAJA\\exercises\\atomic-histogram\_solution.cpp"" exited with code 2. \[C:\\Users\\smith84\\projec
ts\\raja\\clean\\cudabuild\\exercises\\atomic-histogram\_solution.vcxproj]
```

\## Kenny Weiss work on AXOM

Steve : Learned of Kenny branch of AXOM.    Kenny took Steve's CUDA 13 branch and made patches with extra fixes for Windows.   Steve was starting to parse through Kenny patching.   The template issue above might be have been similar to what is patched in "kweiss-windows-cuda-msvc.patch" pulled from Kenny's repository.   Current RAJA has modified the templates so this needs some work.

Steve goes on vacation :(

Kenny/AI found some other changes, note Kenny was using VS 2022 and Cuda 13.

\## Kenny Notes:

Hi Steve,

I do!

(CC-ing Rich and Brian Han since they’re also likely interested in the details)

I’m still working through some of the details, so I haven’t created a PR yet, but the branch is here: https://github.com/llnl/axom/compare/develop...feature/kweiss/windows-gpu

I used codex w/ gpt@5.5 to work through a bunch of the issues. The compilation times on my laptop (6 core i7; I forget the GPU, but it was CUDA\_ARCHITECTURE 75) can be really long, so each round took a few hours, but it eventually got there.

Axom uses vcpkg (https://github.com/microsoft/vcpkg)  through uberenv ( https://github.com/llnl/uberenv/  |  https://uberenv.readthedocs.io/en/latest/)

to manage our TPLs on windows. We manage some local ports of raja, umpire, mfem, conduit, …  inside of axom’s   ./scripts/vcpkg\_ports  directory.

(ports are equivalent to spack package recipes)

To get this working w/ uberenv, I had to add some slight edits: https://github.com/llnl/uberenv/pull/160 -- which should be merged soon

I also have a few changes to BLT, but I haven’t yet looked closely enough to determine which are strictly necessary, and haven’t posted a PR/branch to BLT’s github yet

My branch has a bunch of updates to our vcpkg ports  to get raja, umpire and camp to build on msvc+cuda.

Since my build uses cuda@13, I needed some patches from your raja and camp PRs related to cuda@13.

To set up my build environment on Windows, I had to install:

the cuda toolkit ( https://developer.nvidia.com/cuda-downloads --  I am using cuda@13)

MS visual studio community ( https://visualstudio.microsoft.com/vs/community/  --

I am using visual studio community 2022. Presumably it’d work w/ visual studio community 2026 ).

If you need MPI, you also need to install MS MPI -- https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi

Some gotchas in my msvc+cuda porting:

cuda@13 has some changes (but you’re clearly aware of those)

MSVC does not support extended lambdas inside of if constexpr --

See item 11 in Section 5.3.8.4  -- https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-support.html#extended-lambda-restrictions

Microsoft defines some regular words as macros, e.g.  min, max, small, large

I hope this helps and would be happy to discuss this further/answer questions,

Kenny

\--

Kenneth Weiss

Numerics, Modular Applications and Performance Group Leader

ASQ Division, Computing Directorate, LLNL

