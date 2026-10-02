;
; Input signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
;
; Output signature:
;
; Name                 Index   Mask Register SysValue  Format   Used
; -------------------- ----- ------ -------- -------- ------- ------
; no parameters
; shader hash: 770e908fca393afcffac8497ffb20e11
;
; Pipeline Runtime Information: 
;
;PSVRuntimeInfo:
; Compute Shader
; NumThreads=(8,8,1)
; MinimumExpectedWaveLaneCount: 0
; MaximumExpectedWaveLaneCount: 4294967295
; UsesViewID: false
; SigInputElements: 0
; SigOutputElements: 0
; SigPatchConstOrPrimElements: 0
; SigInputVectors: 0
; SigOutputVectors[0]: 0
; SigOutputVectors[1]: 0
; SigOutputVectors[2]: 0
; SigOutputVectors[3]: 0
; EntryFunctionName: CSConcatCopy
;
;
; Buffer Definitions:
;
; cbuffer pc
; {
;
;   struct pc
;   {
;
;       struct struct.ConcatPushConstants
;       {
;
;           uint h;                                   ; Offset:    0
;           uint w;                                   ; Offset:    4
;           uint src_c4;                              ; Offset:    8
;           uint dst_c4;                              ; Offset:   12
;           uint dst_c4_off;                          ; Offset:   16
;       
;       } pc;                                         ; Offset:    0
;
;   
;   } pc;                                             ; Offset:    0 Size:    20
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
; pc                                cbuffer      NA          NA     CB0            cb0     1
; src                               texture    byte         r/o      T0             t0     1
; dst                                   UAV    byte         r/w      U0             u0     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }
%struct.ByteAddressBuffer = type { i32 }
%struct.RWByteAddressBuffer = type { i32 }
%pc = type { %struct.ConcatPushConstants }
%struct.ConcatPushConstants = type { i32, i32, i32, i32, i32 }

define void @CSConcatCopy() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %5 = call i32 @dx.op.threadId.i32(i32 93, i32 1)  ; ThreadId(component)
  %6 = call i32 @dx.op.threadId.i32(i32 93, i32 2)  ; ThreadId(component)
  %7 = shl i32 %6, 2
  %8 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %9 = extractvalue %dx.types.CBufRet.i32 %8, 1
  %10 = icmp ult i32 %4, %9
  %11 = extractvalue %dx.types.CBufRet.i32 %8, 0
  %12 = icmp ult i32 %5, %11
  %13 = and i1 %10, %12
  br i1 %13, label %14, label %62

; <label>:14                                      ; preds = %0
  %15 = extractvalue %dx.types.CBufRet.i32 %8, 2
  %16 = icmp ult i32 %7, %15
  br i1 %16, label %17, label %62

; <label>:17                                      ; preds = %14
  %18 = add i32 %5, 1
  %19 = add i32 %9, 2
  %20 = mul i32 %19, %18
  %21 = add i32 %4, 1
  %22 = add i32 %20, %21
  %23 = mul i32 %22, %15
  %24 = add i32 %23, %7
  %25 = extractvalue %dx.types.CBufRet.i32 %8, 3
  %26 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %3, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %27 = extractvalue %dx.types.CBufRet.i32 %26, 0
  %28 = mul i32 %22, %25
  %29 = add i32 %27, %7
  %30 = add i32 %29, %28
  %31 = lshr i32 %24, 2
  %32 = shl i32 %31, 4
  %33 = lshr i32 %30, 2
  %34 = shl i32 %33, 4
  %35 = add i32 %7, 4
  %36 = icmp ugt i32 %35, %15
  br i1 %36, label %43, label %37

; <label>:37                                      ; preds = %17
  %38 = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %2, i32 %32, i32 undef)  ; BufferLoad(srv,index,wot)
  %39 = extractvalue %dx.types.ResRet.i32 %38, 0
  %40 = extractvalue %dx.types.ResRet.i32 %38, 1
  %41 = extractvalue %dx.types.ResRet.i32 %38, 2
  %42 = extractvalue %dx.types.ResRet.i32 %38, 3
  call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle %1, i32 %34, i32 undef, i32 %39, i32 %40, i32 %41, i32 %42, i8 15)  ; BufferStore(uav,coord0,coord1,value0,value1,value2,value3,mask)
  br label %62

; <label>:43                                      ; preds = %17
  %44 = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %2, i32 %32, i32 undef)  ; BufferLoad(srv,index,wot)
  %45 = extractvalue %dx.types.ResRet.i32 %44, 0
  %46 = extractvalue %dx.types.ResRet.i32 %44, 1
  %47 = extractvalue %dx.types.ResRet.i32 %44, 2
  %48 = extractvalue %dx.types.ResRet.i32 %44, 3
  %49 = call %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32 68, %dx.types.Handle %1, i32 %34, i32 undef)  ; BufferLoad(srv,index,wot)
  %50 = extractvalue %dx.types.ResRet.i32 %49, 1
  %51 = extractvalue %dx.types.ResRet.i32 %49, 2
  %52 = extractvalue %dx.types.ResRet.i32 %49, 3
  %53 = or i32 %7, 1
  %54 = icmp ult i32 %53, %15
  %55 = select i1 %54, i32 %46, i32 %50
  %56 = or i32 %7, 2
  %57 = icmp ult i32 %56, %15
  %58 = select i1 %57, i32 %47, i32 %51
  %59 = or i32 %7, 3
  %60 = icmp ult i32 %59, %15
  %61 = select i1 %60, i32 %48, i32 %52
  call void @dx.op.bufferStore.i32(i32 69, %dx.types.Handle %1, i32 %34, i32 undef, i32 %45, i32 %55, i32 %58, i32 %61, i8 15)  ; BufferStore(uav,coord0,coord1,value0,value1,value2,value3,mask)
  br label %62

; <label>:62                                      ; preds = %43, %37, %14, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #1

; Function Attrs: nounwind
declare void @dx.op.bufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8) #2

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.i32 @dx.op.bufferLoad.i32(i32, %dx.types.Handle, i32, i32) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!11}

!0 = !{!"dxcoob 1.8.2502.11 (239921522)"}
!1 = !{i32 1, i32 0}
!2 = !{i32 1, i32 8}
!3 = !{!"cs", i32 6, i32 0}
!4 = !{!5, !7, !9, null}
!5 = !{!6}
!6 = !{i32 0, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!7 = !{!8}
!8 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!9 = !{!10}
!10 = !{i32 0, %pc* undef, !"", i32 0, i32 0, i32 1, i32 20, null}
!11 = !{void ()* @CSConcatCopy, !"CSConcatCopy", null, !4, !12}
!12 = !{i32 0, i64 16, i32 4, !13}
!13 = !{i32 8, i32 8, i32 1}
