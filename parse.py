import re,subprocess,pandas as pd,numpy as np
U='/mnt/user-data/uploads/'
files={2021:'33_0b9d91_2021fn.pdf',2022:'33_32ba3a_Best_track_2022.pdf',2023:'33_31869f_BestTrack_2023__ACR_AMR_Modified_1.pdf',2024:'33_82f41b_33_4d5ff5_Best_Track_2024___1_.pdf'}
rowre=re.compile(r'^(?:(\d{2})[./](\d{2})[./](\d{2,4})\s+)?(\d{4})\s+(\d{1,2}\.\d)\s+(\d{2,3}(?:\.\d)?)\s+(?:(\d(?:\.\d)?|-|\d{2})\s+)?(\d{3,4}(?:\.\d)?)\s+(\d{1,3})\s+(\d{1,3})\s+(D|DD|CS|SCS|VSCS|ESCS)\b')
datere=re.compile(r'^(\d{2})[./](\d{2})[./](\d{2,4})\s*$')
rng={'D':(15,30),'DD':(25,35),'CS':(30,50),'SCS':(45,65),'VSCS':(60,92),'ESCS':(85,125)}
rows=[];sid=0
for yr,f in files.items():
    txt=subprocess.run(['pdftotext','-layout',U+f,'-'],capture_output=True,text=True).stdout
    cur=None;prev=None
    for line in txt.split('\n'):
        s=line.strip()
        if re.match(r'^Table\s*\d+',s,re.I):
            sid+=1;cur=None;prev=None;name=s;continue
        if sid==0: continue
        m=datere.match(s)
        if m: 
            d,mo,y=map(int,m.groups());cur=pd.Timestamp(year=yr if y<100 else y if y>=2021 else yr,month=mo,day=d) if True else None
            if cur.year!=yr and not(cur.year==yr+1): cur=cur.replace(year=yr)
            prev=None;continue
        m=rowre.match(s)
        if not m: continue
        d,mo,y,hhmm,lat,lon,ci,ecp,a,b,g=m.groups()
        if d:
            cur=pd.Timestamp(year=yr,month=int(mo),day=int(d));prev=None
        if cur is None: continue
        hh=int(hhmm[:2])
        if prev is not None and hh<prev: cur=cur+pd.Timedelta(days=1)
        prev=hh
        a,b=int(a),int(b)
        msw=max(a,b);dp=min(a,b)
        if not(rng[g][0]<=msw<=rng[g][1]): continue
        rows.append(dict(year=yr,sid=sid,t=cur+pd.Timedelta(hours=hh),lat=float(lat),lon=float(lon),ecp=float(ecp),msw=msw,dp=dp,grade=g))
df=pd.DataFrame(rows).drop_duplicates(['sid','t']).sort_values(['sid','t'])
df.to_csv('besttrack_2021_2024.csv',index=False)
print(len(df),df.sid.nunique(),df.groupby('year').agg(n=('sid','size'),s=('sid','nunique')))
print(df.grade.value_counts())
print(df.groupby('sid').t.agg(['min','max']).head(8))
