import ast
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=Path(__file__).resolve().parent
data=next(ast.literal_eval(x.value) for x in ast.parse((p/'draw_tradeoff_58.py').read_text()).body if isinstance(x,ast.Assign) and any(getattr(t,'id','')=='data' for t in x.targets))
im=Image.new('RGB',(1500,740),'white');d=ImageDraw.Draw(im)
def txt(x,y,s,n=20,c='#23354a',anchor='la',bold=False):
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',n)
    d.text((x,y),s,font=font,fill=c,anchor=anchor)
txt(45,20,'CelebA  |  Four interventions on shared axes',37,bold=True)
txt(45,78,'512 sources per intervention   /   58 of 80 shards   /   Higher and farther right is better',20)
colors=['#6b4b9b','#2876a4','#008165','#c75d12']
for j,(row,c) in enumerate(zip(data,colors)):txt(45+j*119,126,row[0],21,c,bold=True)
txt(560,126,'CA: circles, solid lines      Label-only: diamonds, dashed lines',20)
def axes(x,w,lo,hi,ticks,title):
    y,h=225,358
    txt(x,177,title,24,bold=True)
    def pt(a,b):return x+(a-lo)/(hi-lo)*w,y+(102-b)/34*h
    for v in [70,80,90,100]:
        xx,yy=pt(lo,v);d.line((xx,yy,x+w,yy),fill='#dce3e8',width=2);txt(x-10,yy,str(v),17,anchor='rm')
    d.line((x,y,x,y+h,x+w,y+h),fill='#82919e',width=2)
    for v in ticks:txt(pt(v,68)[0],y+h+8,str(v),17,anchor='ma')
    txt(x+w/2,y+h+38,'Target F1 (%)',20,anchor='ma')
    return pt
def line(points,c,m):
    for (x,y),(u,v) in zip(points,points[1:]):
        if not m:d.line((x,y,u,v),fill=c,width=3);continue
        L=max(1,((u-x)**2+(v-y)**2)**.5)
        for i in range(0,int(L)+1,12):
            a,b=i/L,min(1,(i+7)/L)
            d.line((x+(u-x)*a,y+(v-y)*a,x+(u-x)*b,y+(v-y)*b),fill=c,width=3)
    for x,y in points:
        if m:d.polygon([(x,y-6),(x+6,y),(x,y+6),(x-6,y)],fill=c)
        else:d.ellipse((x-5,y-5,x+5,y+5),fill=c)
pt=axes(85,725,10,103,[20,40,60,80,100],'A. All individual operating points')
for row,c in zip(data,colors):
    for m in [0,1]:line([pt(*z) for z in row[2+m]],c,m)
pt=axes(915,500,42,73,[45,50,55,60,65,70],'B. Equal-weight mean over interventions')
for m,c in [(0,'#226095'),(1,'#bb4924')]:
    arr=[tuple(sum(r[2+m][j][k] for r in data)/4 for k in (0,1)) for j in range(min(len(r[2+m]) for r in data))]
    pts=[pt(*v) for v in arr];line(pts,c,m);print(m,arr)
    for j,(x,y) in enumerate(pts):
        dx,dy=([(0,16),(0,17),(-12,20),(-12,-17),(22,-8)] if not m else [(0,-18),(0,-18)])[j]
        txt(x+dx,y+dy,str([1,2,3,5,7.5][j]),16,c,'mm',True)
txt(45,667,'Vertical axis: mean non-descendant F1 (%). Right: labels denote sampling CFG; only complete four-intervention means shown.',19)
txt(45,701,'Means are descriptive: the same mean target F1 can hide different success rates on individual interventions.',17)
im.save(p/'celeba_tradeoff_overlay58.png')
