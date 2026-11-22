"""Chạy bộ thí nghiệm mặc định và lưu hình/bảng vào outputs/."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cv_bt2.corner import harris_classic, harris_laplace, scale_pyramid, repeatability, top_n, anms, grid_occupancy
from cv_bt2.data.synthetic import checkerboard_image, scale_image, anms_scene, segmentation_scene, gradient_scene, flat_scene
from cv_bt2.segmentation import region_growing_classic, srg, srg_step_frames, seed_sensitivity
from cv_bt2.segmentation.metrics import partition_difference, count_regions
from cv_bt2.visualization import plot_corners, plot_labels

FIG = ROOT / "outputs" / "figures"; TAB = ROOT / "outputs" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

# A1: Harris vs Harris-Laplace
img_a = checkerboard_image((120,120), 6); img_b = scale_image(img_a, 3)
ha = harris_classic(img_a); hb = harris_classic(img_b)
hla = harris_laplace(img_a, scale_pyramid(.25,1.2,10)); hlb = harris_laplace(img_b, scale_pyramid(.25,1.2,16))
tr = lambda r,c: (3*r,3*c)
df_scale = pd.DataFrame([
    {"method":"Harris", "points_1x":len(ha), "points_3x":len(hb), "repeatability":repeatability(ha,hb,tr,img_b.shape,4)},
    {"method":"Harris-Laplace", "points_1x":len(hla), "points_3x":len(hlb), "repeatability":repeatability(hla,hlb,tr,img_b.shape,4)},
])
df_scale.to_csv(TAB/"harris_scale_comparison.csv", index=False, encoding="utf-8-sig")
fig,ax=plt.subplots(2,2,figsize=(10,9)); plot_corners(img_a,ha,ax[0,0],"Harris 1×"); plot_corners(img_b,hb,ax[0,1],"Harris 3×")
plot_corners(img_a,hla,ax[1,0],"Harris–Laplace 1×"); plot_corners(img_b,hlb,ax[1,1],"Harris–Laplace 3×")
plt.tight_layout(); plt.savefig(FIG/"harris_scale_comparison.png",dpi=160); plt.close(fig)

# A2: Top-N vs ANMS
scene = anms_scene(); candidates = harris_classic(scene, sigma_i=1.5, sigma_d=1.0, thresh_rel=.002)
N=100; pt=top_n(candidates,N); pa=anms(candidates,N,.9)
df_anms=pd.DataFrame([
    {"method":"Top-N","N":len(pt),"grid_occupancy_8x8":grid_occupancy(pt,scene.shape,8)},
    {"method":"ANMS","N":len(pa),"grid_occupancy_8x8":grid_occupancy(pa,scene.shape,8)},
])
df_anms.to_csv(TAB/"topn_anms_comparison.csv",index=False,encoding="utf-8-sig")
fig,ax=plt.subplots(1,2,figsize=(12,5)); plot_corners(scene,pt,ax[0],"Top-N"); plot_corners(scene,pa,ax[1],"ANMS")
plt.tight_layout(); plt.savefig(FIG/"topn_anms_comparison.png",dpi=160); plt.close(fig)

# B: RG vs SRG
seg=segmentation_scene(); seeds=[(60,20),(60,90),(60,160)]
rg_stack=region_growing_classic(seg,seeds,25,"stack"); rg_queue=region_growing_classic(seg,seeds,25,"queue")
srg_labels,bound=srg(seg,seeds)
df_seg=pd.DataFrame([
    {"method":"RG Stack","regions":count_regions(rg_stack),"unassigned":int(np.sum(rg_stack<0))},
    {"method":"RG Queue","regions":count_regions(rg_queue),"unassigned":int(np.sum(rg_queue<0))},
    {"method":"SRG","regions":count_regions(srg_labels),"unassigned":int(np.sum(srg_labels<0))},
])
df_seg.to_csv(TAB/"region_growing_srg_comparison.csv",index=False,encoding="utf-8-sig")
fig,ax=plt.subplots(2,2,figsize=(10,8)); ax[0,0].imshow(seg,cmap="gray"); ax[0,0].set_title("Input"); ax[0,0].axis("off")
plot_labels(rg_stack,ax[0,1],"RG Stack"); plot_labels(rg_queue,ax[1,0],"RG Queue"); plot_labels(srg_labels,ax[1,1],"SRG")
plt.tight_layout(); plt.savefig(FIG/"region_growing_srg_comparison.png",dpi=160); plt.close(fig)
plt.figure(figsize=(7,4)); plt.imshow(bound,cmap="gray"); plt.title("SRG boundary mask"); plt.axis("off"); plt.savefig(FIG/"srg_boundary.png",dpi=160,bbox_inches="tight"); plt.close()

# SRG supplementary: Lazy/Frozen, ties, frames, sensitivity
grad=gradient_scene((80,140),True); gseeds=[(40,5),(40,134)]
lazy,_=srg(grad,gseeds,stale="lazy"); frozen,_=srg(grad,gseeds,stale="frozen")
flat=flat_scene(); fseeds=[(50,40),(50,120)]
ties={m:srg(flat,fseeds,tie=m)[0] for m in ("insertion","raster","random")}
frames=srg_step_frames(seg,seeds,n_frames=8)
fig,ax=plt.subplots(2,4,figsize=(12,6))
for i,a in enumerate(ax.flat): plot_labels(frames[i],ax=a,title=f"SRG frame {i+1}")
plt.tight_layout(); plt.savefig(FIG/"srg_growth_frames.png",dpi=150); plt.close(fig)
fig,ax=plt.subplots(1,3,figsize=(12,4))
for a,m in zip(ax,ties): plot_labels(ties[m],ax=a,title=f"Tie: {m}")
plt.tight_layout(); plt.savefig(FIG/"srg_tie_breaking.png",dpi=150); plt.close(fig)
frequency=seed_sensitivity(grad,gseeds,lambda x,s:srg(x,s),n_trials=20,jitter=3)
plt.figure(figsize=(7,4)); plt.imshow(frequency,cmap="inferno",vmin=0,vmax=1); plt.colorbar(label="Tần suất đổi nhãn"); plt.title("Seed sensitivity")
plt.savefig(FIG/"srg_seed_sensitivity.png",dpi=160,bbox_inches="tight"); plt.close()

print("Đã chạy xong thí nghiệm mặc định.")
print("\nHarris scale comparison:\n",df_scale.to_string(index=False))
print("\nTop-N vs ANMS:\n",df_anms.to_string(index=False))
print("\nRegion Growing vs SRG:\n",df_seg.to_string(index=False))
print("\nLazy vs Frozen different pixels:",int(np.count_nonzero(lazy!=frozen)))
print("\nOutputs:")
for folder in (FIG,TAB):
    for path in sorted(folder.iterdir()): print("-",path.relative_to(ROOT))
