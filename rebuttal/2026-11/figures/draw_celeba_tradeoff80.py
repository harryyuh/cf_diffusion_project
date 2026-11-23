from PIL import Image,ImageDraw,ImageFont
from pathlib import Path
im=Image.new('RGB',(1550,1050),'white'); d=ImageDraw.Draw(im)
def text(x,y,s,n=20,color='#243648',anchor='la',bold=False):
    font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial'+(' Bold' if bold else '')+'.ttf',n)
    d.text((x,y),s,font=font,fill=color,anchor=anchor)
C=['#206a99','#bc501d']
A=[[(31.9328,97.1867),(42.7105,99.7375),(46.7213,100),(50.6329,100),(50.9474,100)],[(90.9366,85.3056),(98.8655,91.0209),(99.6743,92.2705),(100,93.0952),(100,94.0334)],[(34.103,76.4211),(38.4058,87.3214),(42.2182,87.5441),(43.1373,89.4928),(43.3393,87.3617)],[(36.859,70.2282),(65.8762,75.0215),(73.7888,73.7312),(78.8095,73.7041),(79.669,75.455)]]
B=[[(39.3013,95.9596),(52.5547,98.7013),(64.3052,99.4764),(78.8274,99.4764),(82.5175,98.7013)],[(94.8518,88.4076),(99.0291,92.2306),(99.3506,95.466),(99.8369,97.3518),(99.6743,98.0745)],[(35.9431,75.4023),(46.9012,76.9265),(51.7073,77.2174),(56.4593,75.2706),(60.4361,72.0072)],[(15.2174,85.5405),(39.9371,89.2118),(48.9614,90.5925),(53.8682,91.7445),(54.4936,92.9957)]]
for arr in [A,B]:
    arr.append([tuple(sum(p)/4 for p in zip(*pts)) for pts in zip(*arr)])
text(65,32,'CelebA   |   Target editing vs. non-descendant preservation',44,bold=True)
text(65,103,'512 sources per intervention   /   DDIM 100   /   80 of 80 shards',25)
text(65,157,'●  Author CA',24,C[0],bold=True); text(275,157,'◆  Label-only',24,C[1],bold=True)
text(540,159,'Sampling CFG: 1 → 2 → 3 → 5 → 7.5     |     Rings mark CFG 3',22)
def panel(i,x,y,title,xs,ys):
    w,h=385,253
    text(x,y-49,title,27,bold=True)
    def pos(a,b):return x+8+(a-xs[0])/(xs[-1]-xs[0])*(w-16),y+h-8-(b-ys[0])/(ys[-1]-ys[0])*(h-16)
    for b in ys:
        yy=pos(xs[0],b)[1];d.line((x,yy,x+w,yy),fill='#dce3e8',width=2);text(x-8,yy,str(b),17,anchor='rm')
    for a in xs:text(pos(a,ys[0])[0],y+h+8,str(a),17,anchor='ma')
    d.line((x,y,x,y+h,x+w,y+h),fill='#81909e',width=2)
    for j,arr in enumerate([A,B]):
        pts=[pos(*p) for p in arr[i]];d.line(pts,fill=C[j],width=4)
        for k,(xx,yy) in enumerate(pts):
            if j:d.polygon([(xx,yy-6),(xx+6,yy),(xx,yy+6),(xx-6,yy)],fill=C[j])
            else:d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=C[j])
            if k==2:d.ellipse((xx-10,yy-10,xx+10,yy+10),outline=C[j],width=2)
panel(0,80,290,'do(Young)',[30,45,60,75,90],[94,96,98,100])
panel(1,585,290,'do(Male)',[90,92,94,96,98,100],[84,88,92,96,100])
panel(2,1090,290,'do(Beard)',[30,40,50,60,65],[70,75,80,85,90])
panel(3,80,695,'do(Bald)',[10,30,50,70,85],[70,75,80,85,90,95])
panel(4,585,695,'Four-intervention mean',[45,55,65,75],[82,85,88,91,94])
text(1085,690,'How to read',28,bold=True)
for j,s in enumerate(['Right: higher target F1.','Up: higher preservation F1.','Each panel uses its own axis range.','','At CFG 3, mean target F1 is similar:','CA 65.60  /  Label-only 66.08.','Preservation: 88.39  /  90.69.']):
    text(1085,750+29*j,s,21)
text(65,1005,'Both axes are F1 (%). Lines connect measured settings, not fitted curves. No confidence intervals shown.',20)
im.save(Path(__file__).with_name('celeba_tradeoff_full80.png'))
