

"""
def mask_to_box(polygon) -> Polygon.bounds:
    mask_polygon = Polygon(shell=polygon)
    area=mask_polygon.centroid
    bb = mask_polygon.bounds
    return bb
l=[]
for shapes in file["shapes"]:
    p=[]
    for poly in shapes["points"]:
        p.append(np.array(poly,dtype=np.float32))
    l.append(p)
print(len(l))
r=np.array(l,dtype=np.float32)
print(r.shape)
r=torch.tensor(r,dtype=torch.float32)"""

"""
points_list = [np.array(shape["points"], dtype=np.float32) for shape in file["shapes"]]
bbs=[]
for i in points_list:
    bb=masks_to_boxes(i)
    print(bb)
    bbs.append(bb)
t=torch.tensor(bbs,dtype=torch.float32)
print(t.shape)
"""