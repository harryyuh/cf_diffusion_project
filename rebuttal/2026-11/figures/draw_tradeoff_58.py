from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
out=Path(__file__).resolve().parent
im=Image.new("RGB",(1540,1020),"white");d=ImageDraw.Draw(im)
def f(n,b=False):return ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if b else '')+'.ttf',n)
def txt(x,y,s,n=18,col='#23354a',anchor='la',b=False):d.text((x,y),s,font=f(n,b),fill=col,anchor=anchor)
ca='#19679b';lo='#bb4e24'
data=[
('Young',(28,70),[(31.93,97.19),(42.71,99.74),(46.72,100),(50.63,100),(50.95,100)],[(39.3,95.96),(52.55,98.7),(64.31,99.48)]),
('Male',(89,101),[(90.94,85.31),(98.87,91.02),(99.67,92.27),(100,93.1),(100,94.03)],[(94.85,88.41),(99.03,92.23)]),
('Beard',(32,50),[(34.1,76.42),(38.41,87.32),(42.22,87.54),(43.14,89.49),(43.34,87.36)],[(35.94,75.4),(46.9,76.93)]),
('Bald',(10,85),[(36.86,70.23),(65.88,75.02),(73.79,73.73),(78.81,73.7),(79.67,75.46)],[(15.22,85.54),(39.94,89.21)])]
txt(55,24,'CelebA  |  Target editing vs. non-descendant preservation',40,b=True)
txt(55,83,'512 paired sources per intervention   /   DDIM 100   /   interim snapshot: 58 of 80 shards',23)
txt(55,132,'●  Author CA',22,ca,b=True);txt(255,132,'◆  Label-only',22,lo,b=True)
txt(490,133,'Numbers = sampling CFG.  Lines connect measured settings in sweep order.',20)
offset=[[(0,18),(0,19),(-10,18),(-16,-17),(21,-17)],[(0,17),(-32,7),(-29,-7),(-24,-16),(14,-16)],[(0,18),(0,-17),(-25,16),(0,-16),(24,12)],[(0,-18),(0,-17),(-10,17),(-1,33),(16,-14)]]
for i,(name,(xl,xh),a,b) in enumerate(data):
    x=90+755*(i%2); y=240+365*(i//2); W=600;H=235
    txt(x,y-50,'do('+name+')',26,b=True)
    def pt(u,v):return (x+(u-xl)*W/(xh-xl),y+(102-v)*H/34)
    for v in [70,80,90,100]:
        _,yy=pt(xl,v);d.line((x,yy,x+W,yy),fill='#dce3e8',width=2);txt(x-10,yy,str(v),17,anchor='rm')
    ticks=[[30,40,50,60,70],[90,92,94,96,98,100],[34,38,42,46,50],[20,40,60,80]][i]
    for v in ticks:
        xx,_=pt(v,68);txt(xx,y+H+9,str(v),17,anchor='ma')
    d.line((x,y,x,y+H,x+W,y+H),fill='#82919e',width=2)
    for arr,col,isca in [(a,ca,True),(b,lo,False)]:
        points=[pt(*p) for p in arr];d.line(points,fill=col,width=3)
        for j,(xx,yy) in enumerate(points):
            if isca:d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=col)
            else:d.polygon([(xx,yy-7),(xx+7,yy),(xx,yy+7),(xx-7,yy)],fill=col)
            dx,dy=offset[i][j] if isca else (0,19)
            txt(xx+dx,yy+dy,str([1,2,3,5,7.5][j]),17,col,'mm',True)
    txt(x+W/2,y+H+39,'Target F1 (%)  →',19,anchor='ma')
txt(55,937,'Vertical axis: mean non-descendant F1 (%) ↑   |   Same vertical scale throughout; horizontal ranges vary by intervention.',19)
txt(55,970,'Only completed 512-source points are shown. No extrapolation or confidence intervals in this draft.',18,'#576779')
im.save(out/'celeba_tradeoff_draft_58.png')
