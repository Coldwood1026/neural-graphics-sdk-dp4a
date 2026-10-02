;
; Note: shader requires additional functionality:
;       64-Bit integer
;
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
; shader hash: 628331e3c21a0f732ca371b8f764a478
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
; EntryFunctionName: CSConv
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
;       struct struct.ConvPushConstants
;       {
;
;           uint in_w;                                ; Offset:    0
;           uint in_c4;                               ; Offset:    4
;           uint out_h;                               ; Offset:    8
;           uint out_w;                               ; Offset:   12
;           uint out_c4;                              ; Offset:   16
;           int pad_top;                              ; Offset:   20
;           int pad_left;                             ; Offset:   24
;           uint stride_h;                            ; Offset:   28
;           uint stride_w;                            ; Offset:   32
;           int out_zp;                               ; Offset:   36
;           uint kh;                                  ; Offset:   40
;           uint kw;                                  ; Offset:   44
;           uint has_lut;                             ; Offset:   48
;           uint dbg;                                 ; Offset:   52
;           uint inH;                                 ; Offset:   56
;           uint inBorded;                            ; Offset:   60
;           uint outBorded;                           ; Offset:   64
;       
;       } pc;                                         ; Offset:    0
;
;   
;   } pc;                                             ; Offset:    0 Size:    68
;
; }
;
;
; Resource Bindings:
;
; Name                                 Type  Format         Dim      ID      HLSL Bind  Count
; ------------------------------ ---------- ------- ----------- ------- -------------- ------
; pc                                cbuffer      NA          NA     CB0            cb0     1
; actIn                             texture    byte         r/o      T0             t0     1
; wPk                               texture    byte         r/o      T1             t1     1
; biasCr                            texture    byte         r/o      T2             t2     1
; mult                              texture    byte         r/o      T3             t3     1
; shiftB                            texture    byte         r/o      T4             t4     1
; lut                               texture    byte         r/o      T5             t5     1
; actOut                                UAV    byte         r/w      U0             u0     1
; dbgBuf                                UAV    byte         r/w      U1             u1     1
;
target datalayout = "e-m:e-p:32:32-i1:32-i8:32-i16:32-i32:32-i64:64-f16:32-f32:32-f64:64-n8:16:32:64"
target triple = "dxil-ms-dx"

%dx.types.Handle = type { i8* }
%dx.types.CBufRet.i32 = type { i32, i32, i32, i32 }
%dx.types.ResRet.i32 = type { i32, i32, i32, i32, i32 }
%struct.ByteAddressBuffer = type { i32 }
%struct.RWByteAddressBuffer = type { i32 }
%pc = type { %struct.ConvPushConstants }
%struct.ConvPushConstants = type { i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32 }

define void @CSConv() {
  %1 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %2 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 1, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %3 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 5, i32 5, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %4 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 4, i32 4, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %5 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 3, i32 3, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %6 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 2, i32 2, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %7 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 1, i32 1, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %8 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 0, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %9 = call %dx.types.Handle @dx.op.createHandle(i32 57, i8 2, i32 0, i32 0, i1 false)  ; CreateHandle(resourceClass,rangeId,index,nonUniformIndex)
  %10 = call i32 @dx.op.threadId.i32(i32 93, i32 0)  ; ThreadId(component)
  %11 = call i32 @dx.op.threadId.i32(i32 93, i32 1)  ; ThreadId(component)
  %12 = call i32 @dx.op.threadId.i32(i32 93, i32 2)  ; ThreadId(component)
  %13 = alloca [4 x i32], align 4
  %14 = alloca [4 x i32], align 4
  %15 = alloca [4 x i32], align 4
  %16 = alloca [4 x i32], align 4
  %17 = alloca [4 x i32], align 4
  %18 = shl i32 %12, 2
  %19 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %20 = extractvalue %dx.types.CBufRet.i32 %19, 2
  %21 = extractvalue %dx.types.CBufRet.i32 %19, 3
  %22 = mul i32 %21, %20
  %23 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %24 = extractvalue %dx.types.CBufRet.i32 %23, 1
  %25 = mul i32 %22, %24
  %26 = shl i32 %12, 4
  %27 = mul i32 %25, %26
  %28 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %29 = extractvalue %dx.types.CBufRet.i32 %28, 0
  %30 = add i32 %18, 4
  %31 = icmp ult i32 %29, %30
  %32 = sub i32 %29, %18
  %33 = select i1 %31, i32 %32, i32 4
  %34 = extractvalue %dx.types.CBufRet.i32 %23, 3
  %35 = icmp ult i32 %10, %34
  %36 = extractvalue %dx.types.CBufRet.i32 %23, 2
  %37 = icmp ult i32 %11, %36
  %38 = and i1 %35, %37
  br i1 %38, label %39, label %682

; <label>:39                                      ; preds = %0
  %40 = icmp ule i32 %29, %18
  %41 = icmp eq i32 %33, 0
  %42 = or i1 %40, %41
  br i1 %42, label %682, label %43

; <label>:43                                      ; preds = %39
  %44 = extractvalue %dx.types.CBufRet.i32 %23, 0
  %45 = add i32 %44, 2
  %46 = mul i32 %45, %24
  %47 = icmp eq i32 %20, 0
  br i1 %47, label %340, label %48

; <label>:48                                      ; preds = %43
  br label %49

; <label>:49                                      ; preds = %318, %48
  %50 = phi i32 [ %319, %318 ], [ 0, %48 ]
  %51 = phi i32 [ %320, %318 ], [ 0, %48 ]
  %52 = phi i32 [ %321, %318 ], [ 0, %48 ]
  %53 = phi i32 [ %322, %318 ], [ 0, %48 ]
  %54 = phi i32 [ %323, %318 ], [ 0, %48 ]
  %55 = phi i32 [ %324, %318 ], [ 0, %48 ]
  %56 = phi i32 [ %325, %318 ], [ 0, %48 ]
  %57 = phi i32 [ %326, %318 ], [ 0, %48 ]
  %58 = phi i32 [ %327, %318 ], [ 0, %48 ]
  %59 = phi i32 [ %328, %318 ], [ 0, %48 ]
  %60 = phi i32 [ %329, %318 ], [ 0, %48 ]
  %61 = phi i32 [ %330, %318 ], [ 0, %48 ]
  %62 = phi i32 [ %331, %318 ], [ 0, %48 ]
  %63 = phi i32 [ %332, %318 ], [ 0, %48 ]
  %64 = phi i32 [ %333, %318 ], [ 0, %48 ]
  %65 = phi i32 [ %334, %318 ], [ 0, %48 ]
  %66 = phi i32 [ %335, %318 ], [ 0, %48 ]
  %67 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %68 = extractvalue %dx.types.CBufRet.i32 %67, 3
  %69 = mul nsw i32 %68, %11
  %70 = extractvalue %dx.types.CBufRet.i32 %67, 1
  %71 = sub nsw i32 %69, %70
  %72 = add nsw i32 %71, %66
  %73 = add nsw i32 %72, 1
  %74 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %75 = extractvalue %dx.types.CBufRet.i32 %74, 3
  %76 = icmp eq i32 %75, 0
  br i1 %76, label %318, label %77

; <label>:77                                      ; preds = %49
  br label %78

; <label>:78                                      ; preds = %298, %77
  %79 = phi i32 [ %299, %298 ], [ %50, %77 ]
  %80 = phi i32 [ %300, %298 ], [ %51, %77 ]
  %81 = phi i32 [ %301, %298 ], [ %52, %77 ]
  %82 = phi i32 [ %302, %298 ], [ %53, %77 ]
  %83 = phi i32 [ %303, %298 ], [ %54, %77 ]
  %84 = phi i32 [ %304, %298 ], [ %55, %77 ]
  %85 = phi i32 [ %305, %298 ], [ %56, %77 ]
  %86 = phi i32 [ %306, %298 ], [ %57, %77 ]
  %87 = phi i32 [ %307, %298 ], [ %58, %77 ]
  %88 = phi i32 [ %308, %298 ], [ %59, %77 ]
  %89 = phi i32 [ %309, %298 ], [ %60, %77 ]
  %90 = phi i32 [ %310, %298 ], [ %61, %77 ]
  %91 = phi i32 [ %311, %298 ], [ %62, %77 ]
  %92 = phi i32 [ %312, %298 ], [ %63, %77 ]
  %93 = phi i32 [ %313, %298 ], [ %64, %77 ]
  %94 = phi i32 [ %314, %298 ], [ %65, %77 ]
  %95 = phi i32 [ %315, %298 ], [ 0, %77 ]
  %96 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %97 = extractvalue %dx.types.CBufRet.i32 %96, 0
  %98 = mul nsw i32 %97, %10
  %99 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %100 = extractvalue %dx.types.CBufRet.i32 %99, 2
  %101 = sub nsw i32 %98, %100
  %102 = add nsw i32 %101, %95
  %103 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %104 = extractvalue %dx.types.CBufRet.i32 %103, 3
  %105 = icmp eq i32 %104, 0
  br i1 %105, label %106, label %126

; <label>:106                                     ; preds = %78
  %107 = icmp slt i32 %72, 0
  br i1 %107, label %117, label %108

; <label>:108                                     ; preds = %106
  %109 = extractvalue %dx.types.CBufRet.i32 %103, 2
  %110 = icmp sge i32 %72, %109
  %111 = icmp slt i32 %102, 0
  %112 = or i1 %110, %111
  br i1 %112, label %117, label %113

; <label>:113                                     ; preds = %108
  %114 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %115 = extractvalue %dx.types.CBufRet.i32 %114, 0
  %116 = icmp sge i32 %102, %115
  br label %117

; <label>:117                                     ; preds = %113, %108, %106
  %118 = phi i1 [ true, %108 ], [ true, %106 ], [ %116, %113 ]
  %119 = zext i1 %118 to i32
  %120 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %121 = extractvalue %dx.types.CBufRet.i32 %120, 0
  %122 = extractvalue %dx.types.CBufRet.i32 %120, 1
  %123 = mul i32 %121, %72
  %124 = add i32 %123, %102
  %125 = mul i32 %124, %122
  br label %133

; <label>:126                                     ; preds = %78
  %127 = add nsw i32 %102, 1
  %128 = mul i32 %46, %73
  %129 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %130 = extractvalue %dx.types.CBufRet.i32 %129, 1
  %131 = mul i32 %130, %127
  %132 = add i32 %131, %128
  br label %133

; <label>:133                                     ; preds = %126, %117
  %134 = phi i32 [ %125, %117 ], [ %132, %126 ]
  %135 = phi i32 [ %119, %117 ], [ 0, %126 ]
  %136 = extractvalue %dx.types.CBufRet.i32 %96, 3
  %137 = mul i32 %136, %66
  %138 = add i32 %137, %95
  %139 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %140 = extractvalue %dx.types.CBufRet.i32 %139, 1
  %141 = mul i32 %140, %138
  %142 = icmp eq i32 %140, 0
  br i1 %142, label %298, label %143

; <label>:143                                     ; preds = %133
  br label %144

; <label>:144                                     ; preds = %288, %143
  %145 = phi i32 [ %175, %288 ], [ %79, %143 ]
  %146 = phi i32 [ %181, %288 ], [ %80, %143 ]
  %147 = phi i32 [ %187, %288 ], [ %81, %143 ]
  %148 = phi i32 [ %193, %288 ], [ %82, %143 ]
  %149 = phi i32 [ %223, %288 ], [ %83, %143 ]
  %150 = phi i32 [ %224, %288 ], [ %84, %143 ]
  %151 = phi i32 [ %225, %288 ], [ %85, %143 ]
  %152 = phi i32 [ %226, %288 ], [ %86, %143 ]
  %153 = phi i32 [ %256, %288 ], [ %87, %143 ]
  %154 = phi i32 [ %257, %288 ], [ %88, %143 ]
  %155 = phi i32 [ %258, %288 ], [ %89, %143 ]
  %156 = phi i32 [ %259, %288 ], [ %90, %143 ]
  %157 = phi i32 [ %289, %288 ], [ %91, %143 ]
  %158 = phi i32 [ %290, %288 ], [ %92, %143 ]
  %159 = phi i32 [ %291, %288 ], [ %93, %143 ]
  %160 = phi i32 [ %292, %288 ], [ %94, %143 ]
  %161 = phi i32 [ %293, %288 ], [ 0, %143 ]
  %162 = icmp eq i32 %135, 0
  br i1 %162, label %163, label %168

; <label>:163                                     ; preds = %144
  %164 = add i32 %161, %134
  %165 = shl i32 %164, 2
  %166 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %8, i32 %165, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %167 = extractvalue %dx.types.ResRet.i32 %166, 0
  br label %168

; <label>:168                                     ; preds = %163, %144
  %169 = phi i32 [ %167, %163 ], [ -2139062144, %144 ]
  %170 = add i32 %161, %141
  %171 = add i32 %170, %27
  %172 = shl i32 %171, 2
  %173 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %172, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %174 = extractvalue %dx.types.ResRet.i32 %173, 0
  %175 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %145, i32 %169, i32 %174)  ; Dot4AddI8Packed(acc,a,b)
  %176 = add i32 %27, %25
  %177 = add i32 %176, %170
  %178 = shl i32 %177, 2
  %179 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %178, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %180 = extractvalue %dx.types.ResRet.i32 %179, 0
  %181 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %146, i32 %169, i32 %180)  ; Dot4AddI8Packed(acc,a,b)
  %182 = shl i32 %25, 1
  %183 = add i32 %171, %182
  %184 = shl i32 %183, 2
  %185 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %184, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %186 = extractvalue %dx.types.ResRet.i32 %185, 0
  %187 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %147, i32 %169, i32 %186)  ; Dot4AddI8Packed(acc,a,b)
  %188 = mul i32 %25, 3
  %189 = add i32 %171, %188
  %190 = shl i32 %189, 2
  %191 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %190, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %192 = extractvalue %dx.types.ResRet.i32 %191, 0
  %193 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %148, i32 %169, i32 %192)  ; Dot4AddI8Packed(acc,a,b)
  %194 = icmp ugt i32 %33, 1
  br i1 %194, label %195, label %222

; <label>:195                                     ; preds = %168
  %196 = shl i32 %25, 2
  %197 = add i32 %161, %141
  %198 = add i32 %197, %196
  %199 = add i32 %198, %27
  %200 = shl i32 %199, 2
  %201 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %200, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %202 = extractvalue %dx.types.ResRet.i32 %201, 0
  %203 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %149, i32 %169, i32 %202)  ; Dot4AddI8Packed(acc,a,b)
  %204 = add i32 %27, %25
  %205 = add i32 %204, %198
  %206 = shl i32 %205, 2
  %207 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %206, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %208 = extractvalue %dx.types.ResRet.i32 %207, 0
  %209 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %150, i32 %169, i32 %208)  ; Dot4AddI8Packed(acc,a,b)
  %210 = shl i32 %25, 1
  %211 = add i32 %199, %210
  %212 = shl i32 %211, 2
  %213 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %212, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %214 = extractvalue %dx.types.ResRet.i32 %213, 0
  %215 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %151, i32 %169, i32 %214)  ; Dot4AddI8Packed(acc,a,b)
  %216 = mul i32 %25, 3
  %217 = add i32 %199, %216
  %218 = shl i32 %217, 2
  %219 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %218, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %220 = extractvalue %dx.types.ResRet.i32 %219, 0
  %221 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %152, i32 %169, i32 %220)  ; Dot4AddI8Packed(acc,a,b)
  br label %222

; <label>:222                                     ; preds = %195, %168
  %223 = phi i32 [ %203, %195 ], [ %149, %168 ]
  %224 = phi i32 [ %209, %195 ], [ %150, %168 ]
  %225 = phi i32 [ %215, %195 ], [ %151, %168 ]
  %226 = phi i32 [ %221, %195 ], [ %152, %168 ]
  %227 = icmp ugt i32 %33, 2
  br i1 %227, label %228, label %255

; <label>:228                                     ; preds = %222
  %229 = shl i32 %25, 3
  %230 = add i32 %161, %141
  %231 = add i32 %230, %229
  %232 = add i32 %231, %27
  %233 = shl i32 %232, 2
  %234 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %233, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %235 = extractvalue %dx.types.ResRet.i32 %234, 0
  %236 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %153, i32 %169, i32 %235)  ; Dot4AddI8Packed(acc,a,b)
  %237 = add i32 %27, %25
  %238 = add i32 %237, %231
  %239 = shl i32 %238, 2
  %240 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %239, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %241 = extractvalue %dx.types.ResRet.i32 %240, 0
  %242 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %154, i32 %169, i32 %241)  ; Dot4AddI8Packed(acc,a,b)
  %243 = shl i32 %25, 1
  %244 = add i32 %232, %243
  %245 = shl i32 %244, 2
  %246 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %245, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %247 = extractvalue %dx.types.ResRet.i32 %246, 0
  %248 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %155, i32 %169, i32 %247)  ; Dot4AddI8Packed(acc,a,b)
  %249 = mul i32 %25, 3
  %250 = add i32 %232, %249
  %251 = shl i32 %250, 2
  %252 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %251, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %253 = extractvalue %dx.types.ResRet.i32 %252, 0
  %254 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %156, i32 %169, i32 %253)  ; Dot4AddI8Packed(acc,a,b)
  br label %255

; <label>:255                                     ; preds = %228, %222
  %256 = phi i32 [ %236, %228 ], [ %153, %222 ]
  %257 = phi i32 [ %242, %228 ], [ %154, %222 ]
  %258 = phi i32 [ %248, %228 ], [ %155, %222 ]
  %259 = phi i32 [ %254, %228 ], [ %156, %222 ]
  %260 = icmp ugt i32 %33, 3
  br i1 %260, label %261, label %288

; <label>:261                                     ; preds = %255
  %262 = mul i32 %25, 12
  %263 = add i32 %161, %141
  %264 = add i32 %263, %262
  %265 = add i32 %264, %27
  %266 = shl i32 %265, 2
  %267 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %266, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %268 = extractvalue %dx.types.ResRet.i32 %267, 0
  %269 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %157, i32 %169, i32 %268)  ; Dot4AddI8Packed(acc,a,b)
  %270 = add i32 %27, %25
  %271 = add i32 %270, %264
  %272 = shl i32 %271, 2
  %273 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %272, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %274 = extractvalue %dx.types.ResRet.i32 %273, 0
  %275 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %158, i32 %169, i32 %274)  ; Dot4AddI8Packed(acc,a,b)
  %276 = shl i32 %25, 1
  %277 = add i32 %265, %276
  %278 = shl i32 %277, 2
  %279 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %278, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %280 = extractvalue %dx.types.ResRet.i32 %279, 0
  %281 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %159, i32 %169, i32 %280)  ; Dot4AddI8Packed(acc,a,b)
  %282 = mul i32 %25, 3
  %283 = add i32 %265, %282
  %284 = shl i32 %283, 2
  %285 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %7, i32 %284, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %286 = extractvalue %dx.types.ResRet.i32 %285, 0
  %287 = call i32 @dx.op.dot4AddPacked.i32(i32 163, i32 %160, i32 %169, i32 %286)  ; Dot4AddI8Packed(acc,a,b)
  br label %288

; <label>:288                                     ; preds = %261, %255
  %289 = phi i32 [ %269, %261 ], [ %157, %255 ]
  %290 = phi i32 [ %275, %261 ], [ %158, %255 ]
  %291 = phi i32 [ %281, %261 ], [ %159, %255 ]
  %292 = phi i32 [ %287, %261 ], [ %160, %255 ]
  %293 = add i32 %161, 1
  %294 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %295 = extractvalue %dx.types.CBufRet.i32 %294, 1
  %296 = icmp ult i32 %293, %295
  br i1 %296, label %144, label %297

; <label>:297                                     ; preds = %288
  br label %298

; <label>:298                                     ; preds = %297, %133
  %299 = phi i32 [ %79, %133 ], [ %175, %297 ]
  %300 = phi i32 [ %80, %133 ], [ %181, %297 ]
  %301 = phi i32 [ %81, %133 ], [ %187, %297 ]
  %302 = phi i32 [ %82, %133 ], [ %193, %297 ]
  %303 = phi i32 [ %83, %133 ], [ %223, %297 ]
  %304 = phi i32 [ %84, %133 ], [ %224, %297 ]
  %305 = phi i32 [ %85, %133 ], [ %225, %297 ]
  %306 = phi i32 [ %86, %133 ], [ %226, %297 ]
  %307 = phi i32 [ %87, %133 ], [ %256, %297 ]
  %308 = phi i32 [ %88, %133 ], [ %257, %297 ]
  %309 = phi i32 [ %89, %133 ], [ %258, %297 ]
  %310 = phi i32 [ %90, %133 ], [ %259, %297 ]
  %311 = phi i32 [ %91, %133 ], [ %289, %297 ]
  %312 = phi i32 [ %92, %133 ], [ %290, %297 ]
  %313 = phi i32 [ %93, %133 ], [ %291, %297 ]
  %314 = phi i32 [ %94, %133 ], [ %292, %297 ]
  %315 = add i32 %95, 1
  %316 = icmp ult i32 %315, %136
  br i1 %316, label %78, label %317

; <label>:317                                     ; preds = %298
  br label %318

; <label>:318                                     ; preds = %317, %49
  %319 = phi i32 [ %50, %49 ], [ %299, %317 ]
  %320 = phi i32 [ %51, %49 ], [ %300, %317 ]
  %321 = phi i32 [ %52, %49 ], [ %301, %317 ]
  %322 = phi i32 [ %53, %49 ], [ %302, %317 ]
  %323 = phi i32 [ %54, %49 ], [ %303, %317 ]
  %324 = phi i32 [ %55, %49 ], [ %304, %317 ]
  %325 = phi i32 [ %56, %49 ], [ %305, %317 ]
  %326 = phi i32 [ %57, %49 ], [ %306, %317 ]
  %327 = phi i32 [ %58, %49 ], [ %307, %317 ]
  %328 = phi i32 [ %59, %49 ], [ %308, %317 ]
  %329 = phi i32 [ %60, %49 ], [ %309, %317 ]
  %330 = phi i32 [ %61, %49 ], [ %310, %317 ]
  %331 = phi i32 [ %62, %49 ], [ %311, %317 ]
  %332 = phi i32 [ %63, %49 ], [ %312, %317 ]
  %333 = phi i32 [ %64, %49 ], [ %313, %317 ]
  %334 = phi i32 [ %65, %49 ], [ %314, %317 ]
  %335 = add i32 %66, 1
  %336 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %337 = extractvalue %dx.types.CBufRet.i32 %336, 2
  %338 = icmp ult i32 %335, %337
  br i1 %338, label %49, label %339

; <label>:339                                     ; preds = %318
  br label %340

; <label>:340                                     ; preds = %339, %43
  %341 = phi i32 [ 0, %43 ], [ %319, %339 ]
  %342 = phi i32 [ 0, %43 ], [ %320, %339 ]
  %343 = phi i32 [ 0, %43 ], [ %321, %339 ]
  %344 = phi i32 [ 0, %43 ], [ %322, %339 ]
  %345 = phi i32 [ 0, %43 ], [ %323, %339 ]
  %346 = phi i32 [ 0, %43 ], [ %324, %339 ]
  %347 = phi i32 [ 0, %43 ], [ %325, %339 ]
  %348 = phi i32 [ 0, %43 ], [ %326, %339 ]
  %349 = phi i32 [ 0, %43 ], [ %327, %339 ]
  %350 = phi i32 [ 0, %43 ], [ %328, %339 ]
  %351 = phi i32 [ 0, %43 ], [ %329, %339 ]
  %352 = phi i32 [ 0, %43 ], [ %330, %339 ]
  %353 = phi i32 [ 0, %43 ], [ %331, %339 ]
  %354 = phi i32 [ 0, %43 ], [ %332, %339 ]
  %355 = phi i32 [ 0, %43 ], [ %333, %339 ]
  %356 = phi i32 [ 0, %43 ], [ %334, %339 ]
  %357 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %358 = extractvalue %dx.types.CBufRet.i32 %357, 1
  %359 = icmp ne i32 %358, 0
  %360 = icmp eq i32 %18, 0
  %361 = and i1 %360, %359
  br i1 %361, label %362, label %420

; <label>:362                                     ; preds = %340
  %363 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %364 = extractvalue %dx.types.CBufRet.i32 %363, 3
  %365 = mul i32 %364, %11
  %366 = add i32 %365, %10
  %367 = mul i32 %366, 24
  %368 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i32 0, i32 0
  store i32 %341, i32* %368, align 4, !tbaa !20
  %369 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i32 0, i32 1
  store i32 %342, i32* %369, align 4, !tbaa !20
  %370 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i32 0, i32 2
  store i32 %343, i32* %370, align 4, !tbaa !20
  %371 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i32 0, i32 3
  store i32 %344, i32* %371, align 4, !tbaa !20
  br label %372

; <label>:372                                     ; preds = %416, %362
  %373 = phi i32 [ %341, %362 ], [ %418, %416 ]
  %374 = phi i32 [ 0, %362 ], [ %414, %416 ]
  %375 = shl nsw i32 %374, 2
  %376 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %6, i32 %375, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %377 = extractvalue %dx.types.ResRet.i32 %376, 0
  %378 = add nsw i32 %377, %373
  %379 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %4, i32 %375, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %380 = extractvalue %dx.types.ResRet.i32 %379, 0
  %381 = sext i32 %378 to i64
  %382 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %375, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %383 = extractvalue %dx.types.ResRet.i32 %382, 0
  %384 = sext i32 %383 to i64
  %385 = mul nsw i64 %384, %381
  %386 = add i32 %380, 63
  %387 = zext i32 %386 to i64
  %388 = and i64 %387, 63
  %389 = shl i64 1, %388
  %390 = add nsw i64 %385, %389
  %391 = zext i32 %380 to i64
  %392 = and i64 %391, 63
  %393 = ashr i64 %390, %392
  %394 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %395 = extractvalue %dx.types.CBufRet.i32 %394, 1
  %396 = sext i32 %395 to i64
  %397 = add nsw i64 %393, %396
  %398 = call i64 @dx.op.binary.i64(i32 37, i64 %397, i64 -128)  ; IMax(a,b)
  %399 = call i64 @dx.op.binary.i64(i32 38, i64 %398, i64 127)  ; IMin(a,b)
  %400 = mul nuw nsw i32 %374, 6
  %401 = add i32 %400, %367
  %402 = shl i32 %401, 2
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %402, i32 undef, i32 %378, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %403 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %375, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %404 = extractvalue %dx.types.ResRet.i32 %403, 0
  %405 = or i32 %402, 4
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %405, i32 undef, i32 %404, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %406 = add i32 %402, 8
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %406, i32 undef, i32 %380, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %407 = trunc i64 %393 to i32
  %408 = add i32 %402, 12
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %408, i32 undef, i32 %407, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %409 = trunc i64 %399 to i32
  %410 = add i32 %402, 16
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %410, i32 undef, i32 %409, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %411 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %412 = extractvalue %dx.types.CBufRet.i32 %411, 1
  %413 = add i32 %402, 20
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %1, i32 %413, i32 undef, i32 %412, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %414 = add nuw nsw i32 %374, 1
  %415 = icmp eq i32 %414, 4
  br i1 %415, label %419, label %416

; <label>:416                                     ; preds = %372
  %417 = getelementptr inbounds [4 x i32], [4 x i32]* %13, i32 0, i32 %414
  %418 = load i32, i32* %417, align 4, !tbaa !20
  br label %372

; <label>:419                                     ; preds = %372
  br label %420

; <label>:420                                     ; preds = %419, %340
  %421 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 4)  ; CBufferLoadLegacy(handle,regIndex)
  %422 = extractvalue %dx.types.CBufRet.i32 %421, 0
  %423 = icmp eq i32 %422, 0
  %424 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 0)  ; CBufferLoadLegacy(handle,regIndex)
  %425 = extractvalue %dx.types.CBufRet.i32 %424, 3
  %426 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 1)  ; CBufferLoadLegacy(handle,regIndex)
  %427 = extractvalue %dx.types.CBufRet.i32 %426, 0
  br i1 %423, label %428, label %431

; <label>:428                                     ; preds = %420
  %429 = mul i32 %425, %11
  %430 = add i32 %429, %10
  br label %437

; <label>:431                                     ; preds = %420
  %432 = add i32 %11, 1
  %433 = add i32 %425, 2
  %434 = mul i32 %433, %432
  %435 = add i32 %10, 1
  %436 = add i32 %435, %434
  br label %437

; <label>:437                                     ; preds = %431, %428
  %438 = phi i32 [ %430, %428 ], [ %436, %431 ]
  %439 = mul i32 %438, %427
  %440 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i32 0, i32 0
  store i32 %341, i32* %440, align 4, !tbaa !20
  %441 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i32 0, i32 1
  store i32 %342, i32* %441, align 4, !tbaa !20
  %442 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i32 0, i32 2
  store i32 %343, i32* %442, align 4, !tbaa !20
  %443 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i32 0, i32 3
  store i32 %344, i32* %443, align 4, !tbaa !20
  br label %444

; <label>:444                                     ; preds = %491, %437
  %445 = phi i32 [ %341, %437 ], [ %493, %491 ]
  %446 = phi i32 [ 0, %437 ], [ %489, %491 ]
  %447 = phi i32 [ 0, %437 ], [ %488, %491 ]
  %448 = add nuw nsw i32 %446, %26
  %449 = shl nsw i32 %448, 2
  %450 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %6, i32 %449, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %451 = extractvalue %dx.types.ResRet.i32 %450, 0
  %452 = add nsw i32 %451, %445
  %453 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %4, i32 %449, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %454 = extractvalue %dx.types.ResRet.i32 %453, 0
  %455 = sext i32 %452 to i64
  %456 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %449, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %457 = extractvalue %dx.types.ResRet.i32 %456, 0
  %458 = sext i32 %457 to i64
  %459 = mul nsw i64 %458, %455
  %460 = add i32 %454, 63
  %461 = zext i32 %460 to i64
  %462 = and i64 %461, 63
  %463 = shl i64 1, %462
  %464 = add nsw i64 %459, %463
  %465 = zext i32 %454 to i64
  %466 = and i64 %465, 63
  %467 = ashr i64 %464, %466
  %468 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %469 = extractvalue %dx.types.CBufRet.i32 %468, 1
  %470 = sext i32 %469 to i64
  %471 = add nsw i64 %467, %470
  %472 = call i64 @dx.op.binary.i64(i32 37, i64 %471, i64 -128)  ; IMax(a,b)
  %473 = call i64 @dx.op.binary.i64(i32 38, i64 %472, i64 127)  ; IMin(a,b)
  %474 = trunc i64 %473 to i32
  %475 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %476 = extractvalue %dx.types.CBufRet.i32 %475, 0
  %477 = icmp eq i32 %476, 0
  br i1 %477, label %483, label %478

; <label>:478                                     ; preds = %444
  %479 = shl i32 %474, 2
  %480 = add i32 %479, 512
  %481 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %3, i32 %480, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %482 = extractvalue %dx.types.ResRet.i32 %481, 0
  br label %483

; <label>:483                                     ; preds = %478, %444
  %484 = phi i32 [ %482, %478 ], [ %474, %444 ]
  %485 = and i32 %484, 255
  %486 = shl i32 %446, 3
  %487 = shl i32 %485, %486
  %488 = or i32 %487, %447
  %489 = add nuw nsw i32 %446, 1
  %490 = icmp eq i32 %489, 4
  br i1 %490, label %494, label %491

; <label>:491                                     ; preds = %483
  %492 = getelementptr inbounds [4 x i32], [4 x i32]* %14, i32 0, i32 %489
  %493 = load i32, i32* %492, align 4, !tbaa !20
  br label %444

; <label>:494                                     ; preds = %483
  %495 = add i32 %439, %18
  %496 = shl i32 %495, 2
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 %496, i32 undef, i32 %488, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  %497 = icmp ugt i32 %33, 1
  br i1 %497, label %498, label %558

; <label>:498                                     ; preds = %494
  %499 = getelementptr inbounds [4 x i32], [4 x i32]* %15, i32 0, i32 0
  store i32 %345, i32* %499, align 4, !tbaa !20
  %500 = getelementptr inbounds [4 x i32], [4 x i32]* %15, i32 0, i32 1
  store i32 %346, i32* %500, align 4, !tbaa !20
  %501 = getelementptr inbounds [4 x i32], [4 x i32]* %15, i32 0, i32 2
  store i32 %347, i32* %501, align 4, !tbaa !20
  %502 = getelementptr inbounds [4 x i32], [4 x i32]* %15, i32 0, i32 3
  store i32 %348, i32* %502, align 4, !tbaa !20
  %503 = or i32 %26, 4
  br label %504

; <label>:504                                     ; preds = %551, %498
  %505 = phi i32 [ %345, %498 ], [ %553, %551 ]
  %506 = phi i32 [ 0, %498 ], [ %549, %551 ]
  %507 = phi i32 [ 0, %498 ], [ %548, %551 ]
  %508 = add nuw nsw i32 %506, %503
  %509 = shl nsw i32 %508, 2
  %510 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %6, i32 %509, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %511 = extractvalue %dx.types.ResRet.i32 %510, 0
  %512 = add nsw i32 %511, %505
  %513 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %4, i32 %509, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %514 = extractvalue %dx.types.ResRet.i32 %513, 0
  %515 = sext i32 %512 to i64
  %516 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %509, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %517 = extractvalue %dx.types.ResRet.i32 %516, 0
  %518 = sext i32 %517 to i64
  %519 = mul nsw i64 %518, %515
  %520 = add i32 %514, 63
  %521 = zext i32 %520 to i64
  %522 = and i64 %521, 63
  %523 = shl i64 1, %522
  %524 = add nsw i64 %519, %523
  %525 = zext i32 %514 to i64
  %526 = and i64 %525, 63
  %527 = ashr i64 %524, %526
  %528 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %529 = extractvalue %dx.types.CBufRet.i32 %528, 1
  %530 = sext i32 %529 to i64
  %531 = add nsw i64 %527, %530
  %532 = call i64 @dx.op.binary.i64(i32 37, i64 %531, i64 -128)  ; IMax(a,b)
  %533 = call i64 @dx.op.binary.i64(i32 38, i64 %532, i64 127)  ; IMin(a,b)
  %534 = trunc i64 %533 to i32
  %535 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %536 = extractvalue %dx.types.CBufRet.i32 %535, 0
  %537 = icmp eq i32 %536, 0
  br i1 %537, label %543, label %538

; <label>:538                                     ; preds = %504
  %539 = shl i32 %534, 2
  %540 = add i32 %539, 512
  %541 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %3, i32 %540, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %542 = extractvalue %dx.types.ResRet.i32 %541, 0
  br label %543

; <label>:543                                     ; preds = %538, %504
  %544 = phi i32 [ %542, %538 ], [ %534, %504 ]
  %545 = and i32 %544, 255
  %546 = shl i32 %506, 3
  %547 = shl i32 %545, %546
  %548 = or i32 %547, %507
  %549 = add nuw nsw i32 %506, 1
  %550 = icmp eq i32 %549, 4
  br i1 %550, label %554, label %551

; <label>:551                                     ; preds = %543
  %552 = getelementptr inbounds [4 x i32], [4 x i32]* %15, i32 0, i32 %549
  %553 = load i32, i32* %552, align 4, !tbaa !20
  br label %504

; <label>:554                                     ; preds = %543
  %555 = add i32 %439, %18
  %556 = shl i32 %555, 2
  %557 = add i32 %556, 4
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 %557, i32 undef, i32 %548, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %558

; <label>:558                                     ; preds = %554, %494
  %559 = icmp ugt i32 %33, 2
  br i1 %559, label %560, label %620

; <label>:560                                     ; preds = %558
  %561 = getelementptr inbounds [4 x i32], [4 x i32]* %16, i32 0, i32 0
  store i32 %349, i32* %561, align 4, !tbaa !20
  %562 = getelementptr inbounds [4 x i32], [4 x i32]* %16, i32 0, i32 1
  store i32 %350, i32* %562, align 4, !tbaa !20
  %563 = getelementptr inbounds [4 x i32], [4 x i32]* %16, i32 0, i32 2
  store i32 %351, i32* %563, align 4, !tbaa !20
  %564 = getelementptr inbounds [4 x i32], [4 x i32]* %16, i32 0, i32 3
  store i32 %352, i32* %564, align 4, !tbaa !20
  %565 = or i32 %26, 8
  br label %566

; <label>:566                                     ; preds = %613, %560
  %567 = phi i32 [ %349, %560 ], [ %615, %613 ]
  %568 = phi i32 [ 0, %560 ], [ %611, %613 ]
  %569 = phi i32 [ 0, %560 ], [ %610, %613 ]
  %570 = add nuw nsw i32 %568, %565
  %571 = shl nsw i32 %570, 2
  %572 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %6, i32 %571, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %573 = extractvalue %dx.types.ResRet.i32 %572, 0
  %574 = add nsw i32 %573, %567
  %575 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %4, i32 %571, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %576 = extractvalue %dx.types.ResRet.i32 %575, 0
  %577 = sext i32 %574 to i64
  %578 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %571, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %579 = extractvalue %dx.types.ResRet.i32 %578, 0
  %580 = sext i32 %579 to i64
  %581 = mul nsw i64 %580, %577
  %582 = add i32 %576, 63
  %583 = zext i32 %582 to i64
  %584 = and i64 %583, 63
  %585 = shl i64 1, %584
  %586 = add nsw i64 %581, %585
  %587 = zext i32 %576 to i64
  %588 = and i64 %587, 63
  %589 = ashr i64 %586, %588
  %590 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %591 = extractvalue %dx.types.CBufRet.i32 %590, 1
  %592 = sext i32 %591 to i64
  %593 = add nsw i64 %589, %592
  %594 = call i64 @dx.op.binary.i64(i32 37, i64 %593, i64 -128)  ; IMax(a,b)
  %595 = call i64 @dx.op.binary.i64(i32 38, i64 %594, i64 127)  ; IMin(a,b)
  %596 = trunc i64 %595 to i32
  %597 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %598 = extractvalue %dx.types.CBufRet.i32 %597, 0
  %599 = icmp eq i32 %598, 0
  br i1 %599, label %605, label %600

; <label>:600                                     ; preds = %566
  %601 = shl i32 %596, 2
  %602 = add i32 %601, 512
  %603 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %3, i32 %602, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %604 = extractvalue %dx.types.ResRet.i32 %603, 0
  br label %605

; <label>:605                                     ; preds = %600, %566
  %606 = phi i32 [ %604, %600 ], [ %596, %566 ]
  %607 = and i32 %606, 255
  %608 = shl i32 %568, 3
  %609 = shl i32 %607, %608
  %610 = or i32 %609, %569
  %611 = add nuw nsw i32 %568, 1
  %612 = icmp eq i32 %611, 4
  br i1 %612, label %616, label %613

; <label>:613                                     ; preds = %605
  %614 = getelementptr inbounds [4 x i32], [4 x i32]* %16, i32 0, i32 %611
  %615 = load i32, i32* %614, align 4, !tbaa !20
  br label %566

; <label>:616                                     ; preds = %605
  %617 = add i32 %439, %18
  %618 = shl i32 %617, 2
  %619 = add i32 %618, 8
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 %619, i32 undef, i32 %610, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %620

; <label>:620                                     ; preds = %616, %558
  %621 = icmp ugt i32 %33, 3
  br i1 %621, label %622, label %682

; <label>:622                                     ; preds = %620
  %623 = getelementptr inbounds [4 x i32], [4 x i32]* %17, i32 0, i32 0
  store i32 %353, i32* %623, align 4, !tbaa !20
  %624 = getelementptr inbounds [4 x i32], [4 x i32]* %17, i32 0, i32 1
  store i32 %354, i32* %624, align 4, !tbaa !20
  %625 = getelementptr inbounds [4 x i32], [4 x i32]* %17, i32 0, i32 2
  store i32 %355, i32* %625, align 4, !tbaa !20
  %626 = getelementptr inbounds [4 x i32], [4 x i32]* %17, i32 0, i32 3
  store i32 %356, i32* %626, align 4, !tbaa !20
  %627 = or i32 %26, 12
  br label %628

; <label>:628                                     ; preds = %675, %622
  %629 = phi i32 [ %353, %622 ], [ %677, %675 ]
  %630 = phi i32 [ 0, %622 ], [ %673, %675 ]
  %631 = phi i32 [ 0, %622 ], [ %672, %675 ]
  %632 = add nsw i32 %630, %627
  %633 = shl nsw i32 %632, 2
  %634 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %6, i32 %633, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %635 = extractvalue %dx.types.ResRet.i32 %634, 0
  %636 = add nsw i32 %635, %629
  %637 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %4, i32 %633, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %638 = extractvalue %dx.types.ResRet.i32 %637, 0
  %639 = sext i32 %636 to i64
  %640 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %5, i32 %633, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %641 = extractvalue %dx.types.ResRet.i32 %640, 0
  %642 = sext i32 %641 to i64
  %643 = mul nsw i64 %642, %639
  %644 = add i32 %638, 63
  %645 = zext i32 %644 to i64
  %646 = and i64 %645, 63
  %647 = shl i64 1, %646
  %648 = add nsw i64 %643, %647
  %649 = zext i32 %638 to i64
  %650 = and i64 %649, 63
  %651 = ashr i64 %648, %650
  %652 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 2)  ; CBufferLoadLegacy(handle,regIndex)
  %653 = extractvalue %dx.types.CBufRet.i32 %652, 1
  %654 = sext i32 %653 to i64
  %655 = add nsw i64 %651, %654
  %656 = call i64 @dx.op.binary.i64(i32 37, i64 %655, i64 -128)  ; IMax(a,b)
  %657 = call i64 @dx.op.binary.i64(i32 38, i64 %656, i64 127)  ; IMin(a,b)
  %658 = trunc i64 %657 to i32
  %659 = call %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32 59, %dx.types.Handle %9, i32 3)  ; CBufferLoadLegacy(handle,regIndex)
  %660 = extractvalue %dx.types.CBufRet.i32 %659, 0
  %661 = icmp eq i32 %660, 0
  br i1 %661, label %667, label %662

; <label>:662                                     ; preds = %628
  %663 = shl i32 %658, 2
  %664 = add i32 %663, 512
  %665 = call %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32 139, %dx.types.Handle %3, i32 %664, i32 undef, i8 1, i32 4)  ; RawBufferLoad(srv,index,elementOffset,mask,alignment)
  %666 = extractvalue %dx.types.ResRet.i32 %665, 0
  br label %667

; <label>:667                                     ; preds = %662, %628
  %668 = phi i32 [ %666, %662 ], [ %658, %628 ]
  %669 = and i32 %668, 255
  %670 = shl i32 %630, 3
  %671 = shl i32 %669, %670
  %672 = or i32 %671, %631
  %673 = add nuw nsw i32 %630, 1
  %674 = icmp eq i32 %673, 4
  br i1 %674, label %678, label %675

; <label>:675                                     ; preds = %667
  %676 = getelementptr inbounds [4 x i32], [4 x i32]* %17, i32 0, i32 %673
  %677 = load i32, i32* %676, align 4, !tbaa !20
  br label %628

; <label>:678                                     ; preds = %667
  %679 = add i32 %439, %18
  %680 = shl i32 %679, 2
  %681 = add i32 %680, 12
  call void @dx.op.rawBufferStore.i32(i32 140, %dx.types.Handle %2, i32 %681, i32 undef, i32 %672, i32 undef, i32 undef, i32 undef, i8 1, i32 4)  ; RawBufferStore(uav,index,elementOffset,value0,value1,value2,value3,mask,alignment)
  br label %682

; <label>:682                                     ; preds = %678, %620, %39, %0
  ret void
}

; Function Attrs: nounwind readnone
declare i32 @dx.op.threadId.i32(i32, i32) #0

; Function Attrs: nounwind readonly
declare %dx.types.ResRet.i32 @dx.op.rawBufferLoad.i32(i32, %dx.types.Handle, i32, i32, i8, i32) #1

; Function Attrs: nounwind readnone
declare i32 @dx.op.dot4AddPacked.i32(i32, i32, i32, i32) #0

; Function Attrs: nounwind readnone
declare i64 @dx.op.binary.i64(i32, i64, i64) #0

; Function Attrs: nounwind
declare void @dx.op.rawBufferStore.i32(i32, %dx.types.Handle, i32, i32, i32, i32, i32, i32, i8, i32) #2

; Function Attrs: nounwind readonly
declare %dx.types.CBufRet.i32 @dx.op.cbufferLoadLegacy.i32(i32, %dx.types.Handle, i32) #1

; Function Attrs: nounwind readonly
declare %dx.types.Handle @dx.op.createHandle(i32, i8, i32, i32, i1) #1

attributes #0 = { nounwind readnone }
attributes #1 = { nounwind readonly }
attributes #2 = { nounwind }

!llvm.ident = !{!0}
!dx.version = !{!1}
!dx.valver = !{!2}
!dx.shaderModel = !{!3}
!dx.resources = !{!4}
!dx.entryPoints = !{!17}

!0 = !{!"dxcoob 1.8.2502.11 (239921522)"}
!1 = !{i32 1, i32 4}
!2 = !{i32 1, i32 8}
!3 = !{!"cs", i32 6, i32 4}
!4 = !{!5, !12, !15, null}
!5 = !{!6, !7, !8, !9, !10, !11}
!6 = !{i32 0, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i32 0, null}
!7 = !{i32 1, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 1, i32 1, i32 11, i32 0, null}
!8 = !{i32 2, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 2, i32 1, i32 11, i32 0, null}
!9 = !{i32 3, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 3, i32 1, i32 11, i32 0, null}
!10 = !{i32 4, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 4, i32 1, i32 11, i32 0, null}
!11 = !{i32 5, %struct.ByteAddressBuffer* undef, !"", i32 0, i32 5, i32 1, i32 11, i32 0, null}
!12 = !{!13, !14}
!13 = !{i32 0, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 0, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!14 = !{i32 1, %struct.RWByteAddressBuffer* undef, !"", i32 0, i32 1, i32 1, i32 11, i1 false, i1 false, i1 false, null}
!15 = !{!16}
!16 = !{i32 0, %pc* undef, !"", i32 0, i32 0, i32 1, i32 68, null}
!17 = !{void ()* @CSConv, !"CSConv", null, !4, !18}
!18 = !{i32 0, i64 1048592, i32 4, !19}
!19 = !{i32 8, i32 8, i32 1}
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !22, i64 0}
!22 = !{!"omnipotent char", !23, i64 0}
!23 = !{!"Simple C/C++ TBAA"}
