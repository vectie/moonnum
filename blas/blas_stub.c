#include <moonbit.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _WIN32
#include <windows.h>
typedef HMODULE Library;
#define OPEN_LIBRARY(path) LoadLibraryA(path)
#define FIND_SYMBOL(handle,name) GetProcAddress(handle,name)
#define CLOSE_LIBRARY(handle) FreeLibrary(handle)
#else
#include <dlfcn.h>
typedef void *Library;
#define OPEN_LIBRARY(path) dlopen(path,RTLD_NOW|RTLD_LOCAL)
#define FIND_SYMBOL(handle,name) dlsym(handle,name)
#define CLOSE_LIBRARY(handle) dlclose(handle)
#endif

typedef void (*Dgemm)(int,int,int,int,int,int,double,const double *,int,const double *,int,double,double *,int);
typedef void (*Sgemm)(int,int,int,int,int,int,float,const float *,int,const float *,int,float,float *,int);
typedef struct {Library library;Dgemm dgemm;Sgemm sgemm;char error[512];} Blas;

static void destroy_blas(void *raw) {
  Blas *blas=(Blas *)raw;
  if(blas->library) CLOSE_LIBRARY(blas->library);
}

MOONBIT_FFI_EXPORT Blas *moonnum_blas_load(const uint8_t *path) {
  Blas *blas=(Blas *)moonbit_make_external_object(destroy_blas,sizeof(Blas));
  memset(blas,0,sizeof(Blas));
  blas->library=OPEN_LIBRARY((const char *)path);
  if(!blas->library) {
#ifdef _WIN32
    snprintf(blas->error,sizeof(blas->error),"LoadLibrary failed (%lu)",(unsigned long)GetLastError());
#else
    const char *error=dlerror();snprintf(blas->error,sizeof(blas->error),"%s",error?error:"dlopen failed");
#endif
    return blas;
  }
  /* Only the declared LP64 CBLAS ABI is supported; ILP64 symbols are not selected. */
  blas->dgemm=(Dgemm)FIND_SYMBOL(blas->library,"cblas_dgemm");
  blas->sgemm=(Sgemm)FIND_SYMBOL(blas->library,"cblas_sgemm");
  if(!blas->dgemm||!blas->sgemm) {
    snprintf(blas->error,sizeof(blas->error),"library must export LP64 cblas_dgemm and cblas_sgemm");
    CLOSE_LIBRARY(blas->library);blas->library=0;blas->dgemm=0;blas->sgemm=0;
  }
  return blas;
}

MOONBIT_FFI_EXPORT moonbit_bytes_t moonnum_blas_error(Blas *blas) {
  int32_t size=(int32_t)strlen(blas->error);moonbit_bytes_t bytes=moonbit_make_bytes(size,0);memcpy(bytes,blas->error,(size_t)size);return bytes;
}

MOONBIT_FFI_EXPORT void moonnum_blas_dgemm(Blas *blas,double *a,double *b,double *c,int32_t m,int32_t n,int32_t k,int32_t ta,int32_t tb) {
  blas->dgemm(101,ta?112:111,tb?112:111,m,n,k,1.0,a,ta?m:k,b,tb?k:n,0.0,c,n);
}
MOONBIT_FFI_EXPORT void moonnum_blas_sgemm(Blas *blas,float *a,float *b,float *c,int32_t m,int32_t n,int32_t k,int32_t ta,int32_t tb) {
  blas->sgemm(101,ta?112:111,tb?112:111,m,n,k,1.0f,a,ta?m:k,b,tb?k:n,0.0f,c,n);
}
