# minimalLER
A real-time Deferred DrawIndexedIndirect Scene Renderer  
with Compute Frustum Culling (GPU), developed with Vulkan-Hpp.

![ler](https://github.com/Loulfy/minimalLER/blob/main/assets/minimalLER.png?raw=true)

## UI
ZQSD: Camera movement  
E/Shift: Camera Up/Down  
Space: Lock/Unlock mouse  
Ctrl: Unselect light  
Escape: Quit

## Build
You need CMake, Conan 2 and VulkanSDK 1.3 (261) to compile.  
Contains submodules, clone with recursive:
```bash  
git clone --recursive https://github.com/Loulfy/minimalLER
```  

### Windows (tested with Visual Studio Community 2022)

```powershell  
conan install .. --build=missing
cmake -G "Visual Studio 17" .. -DCMAKE_TOOLCHAIN_FILE=path\to\conan_toolchain.cmake -DCMAKE_POLICY_DEFAULT_CMP0091=NEW
cmake --build .
```  

### GNU/Linux (tested with Clang 14)

Conan profile:
```bash  
compiler=clang
compiler.libcxx=libc++
compiler.version=14
```  
Build:
```bash  
export C=/usr/bin/clang
export CXX=/usr/bin/clang++
conan install .. --build=missing
cmake ..
make
```
