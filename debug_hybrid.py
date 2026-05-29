import json, cv2, numpy as np

with open("calibration.json") as f:
    cal = json.load(f)
roi = cal["roi"]
x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
TAPE_EDGE_COLS=10; TAPE_COL_OFFSET=0
col_slice=slice(TAPE_COL_OFFSET,TAPE_COL_OFFSET+TAPE_EDGE_COLS)
BASELINE_FRAMES=100; AIR_REF_ROWS=30
search_start=int(0.25*h); search_end=int(0.95*h)

cap=cv2.VideoCapture("PhaseII_TestD_0003_c4_01.MP4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
bl_frames=[]
for fn in range(BASELINE_FRAMES):
    ret,frame=cap.read()
    rg=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    bl_frames.append(rg[:,col_slice])
bp=np.median(bl_frames,axis=0).astype(np.float32)
air_ref=np.median(bp[:AIR_REF_ROWS])

for fn in [400, 430, 1000]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
    ret,frame=cap.read()
    rg=cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2GRAY)
    sub=rg[:,col_slice].astype(np.float32)
    air_frame=np.median(sub[:AIR_REF_ROWS])
    if air_ref>0 and air_frame>0:
        sub*=air_ref/air_frame
    
    diff=np.median(np.abs(sub-bp),axis=1)
    k=np.ones(51)/51
    dsm=np.convolve(diff,k,mode='same')
    air_offset=np.median(dsm[:AIR_REF_ROWS*4])
    dsm_norm=np.maximum(dsm-air_offset,0)
    
    reg=dsm_norm[search_start:search_end]
    
    print(f"Frame {fn}:")
    print(f"  air_ref={air_ref:.1f}, air_frame={air_frame:.1f}")
    print(f"  air_offset={air_offset:.1f}")
    print(f"  peak dsm={np.max(dsm):.1f}, peak norm={np.max(reg):.1f}")
    
    # Find the gradient
    grads=np.abs(np.diff(reg))
    if np.max(reg)>=4:
        argmax=search_start+np.argmax(grads)
        print(f"  grad argmax: row={argmax}, height={(search_end-argmax)/77.33:.3f}")
    
    # Print the diff_norm profile around important areas
    print(f"  Profile (rows 200-450, where dsm_norm > 5):")
    for r in range(search_start, search_start+250):
        if dsm_norm[r] > 5:
            print(f"    row {r}: diff={diff[r]:.1f}, dsm={dsm[r]:.1f}, dsm_norm={dsm_norm[r]:.1f}, sub={sub[r].mean():.1f}, bl={bp[r].mean():.1f}")
    print()
