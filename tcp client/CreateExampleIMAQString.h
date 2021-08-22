#include "extcode.h"
#pragma pack(push)
#pragma pack(1)

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * CreateExampleIMAQFlattenedString
 */
void __stdcall CreateExampleIMAQFlattenedString(char DataString[], 
	int32_t len);

MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

void __cdecl SetExecuteVIsInPrivateExecutionSystem(Bool32 value);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma pack(pop)

