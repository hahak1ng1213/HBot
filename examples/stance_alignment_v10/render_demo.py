"""Render the saved, validated physics rollout without resimulating it."""
import os
os.environ.update(MUJOCO_GL='egl',MUJOCO_EGL_DEVICE_ID='1',CUDA_VISIBLE_DEVICES='GPU-2a18a801-bbc4-8c59-9f5f-ce5839904141',OMP_NUM_THREADS='1')
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),'/home/zhanghan/tools/hbot-video']
import numpy as np,mujoco,imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
from holoboxing.data.contract import CompetitionAssets
from holoboxing.data.robot import load_scene_model

source=ROOT/'eval-v2';out=ROOT/'demo-v2';out.mkdir(exist_ok=False)
z=np.load(source/'states.npz');metrics=json.loads((source/'metrics.json').read_text());report=json.loads((source/'report.json').read_text())
assert report['state']=='completed' and report['simulation_seconds']==60 and len(z['qpos'])==3000
model=load_scene_model(CompetitionAssets.load(Path('/home/zhanghan/datasets/hbot-assets')))
model.vis.global_.offwidth=960;model.vis.global_.offheight=640
data=mujoco.MjData(model);renderer=mujoco.Renderer(model,width=960,height=640)
cam=mujoco.MjvCamera();cam.distance=4.7;cam.azimuth=135;cam.elevation=-52
roots=[model.body('p1_pelvis').id,model.body('p2_pelvis').id]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
writer=imageio_ffmpeg.write_frames(str(out/'motion.mp4'),(960,640),fps=25,codec='libx264',quality=8,output_params=['-threads','1','-movflags','+faststart']);writer.send(None)
slow=imageio_ffmpeg.write_frames(str(out/'hit-slow.mp4'),(960,640),fps=12.5,codec='libx264',quality=8,output_params=['-threads','1','-movflags','+faststart']);slow.send(None)
stills=[];hits=[];start=time.monotonic()
try:
 for f in range(3000):
    if (f+1)%2 and not 1519<=f<=1589:continue
    data.qpos[:]=z['qpos'][f];data.qvel[:]=z['qvel'][f];mujoco.mj_forward(model,data)
    cam.lookat[:]=data.xpos[roots].mean(0);cam.lookat[2]=.7
    renderer.update_scene(data,camera=cam);pic=Image.fromarray(renderer.render());draw=ImageDraw.Draw(pic)
    m=metrics[f];a=m['alignment'];t=(f+1)*.02
    draw.rectangle((0,0,960,83),fill=(15,18,23))
    text=[f'Stance alignment v10 | saved physics | {t:.2f}s | RED {int(t>=31.17)} : BLUE 0',
          f"{m['skill']} | pelvis {a['pelvis_error_deg']:+.1f}deg | feet {a['feet_error_deg']:+.1f}deg | {m['distance_m']:.2f}m",
          'Attack gate: pelvis <=8deg / feet <=12deg | standard five-point controller']
    for j,line in enumerate(text):draw.text((12,6+j*24),line,font=font,fill='white')
    if 31.17<=t<=32.17:
        draw.rectangle((10,590,740,628),fill=(14,85,42));draw.text((20,598),'VALID HIT: RED right straight -> torso | 20ms impulse 5.191 N.s',font=font,fill='white')
    if (f+1)%2==0:writer.send(np.asarray(pic))
    if 1519<=f<=1589:slow.send(np.asarray(pic))
    if f in [49,149,299,599,999,1559,2099,2999]:
        pic.save(out/f'frame-{t:05.2f}.png');stills.append(pic.resize((480,320)))
    if f in [1553,1557,1558,1559,1561,1565]:hits.append(pic.resize((480,320)))
    if (f+1)%500==0:print('RENDER',t,flush=True)
finally:writer.close();slow.close();renderer.close()
for name,images,cols in [('overview.jpg',stills,2),('hit-review.jpg',hits,3)]:
    sheet=Image.new('RGB',(480*cols,320*((len(images)+cols-1)//cols)))
    for i,pic in enumerate(images):sheet.paste(pic,((i%cols)*480,(i//cols)*320))
    sheet.save(out/name,quality=95)
frames,seconds=imageio_ffmpeg.count_frames_and_secs(str(out/'motion.mp4'));assert frames==1500 and abs(seconds-60)<.01
result=dict(state='completed',frames=frames,seconds=seconds,source='eval-v2/states.npz',
    source_sha256=hashlib.sha256((source/'states.npz').read_bytes()).hexdigest(),
    video_sha256=hashlib.sha256((out/'motion.mp4').read_bytes()).hexdigest(),
    resimulated=False,slow_motion='30.40-31.80s, quarter speed',wall_seconds=time.monotonic()-start)
(out/'video-report.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
