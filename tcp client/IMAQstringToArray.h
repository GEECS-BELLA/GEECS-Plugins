#include "extcode.h"
#pragma pack(push)
#pragma pack(1)

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	int32_t dimSizes[2];
	uint16_t elt[1];
} Uint16ArrayBase;
typedef Uint16ArrayBase **Uint16Array;

/*!
 * IMAQFlattenedStringToArray
 */
void __stdcall IMAQFlattenedStringToArray(char FlattenedImageString[], 
	Uint16Array *ImagePixelsU16);

MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

/*
* Memory Allocation/Resize/Deallocation APIs for type 'Uint16Array'
*/
Uint16Array __cdecl AllocateUint16Array (int32 *dimSizeArr);
MgErr __cdecl ResizeUint16Array (Uint16Array *hdlPtr, int32 *dimSizeArr);
MgErr __cdecl DeAllocateUint16Array (Uint16Array *hdlPtr);

void __cdecl SetExecuteVIsInPrivateExecutionSystem(Bool32 value);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma pack(pop)

