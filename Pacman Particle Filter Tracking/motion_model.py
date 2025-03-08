def manhattanweights(g1,g2,g3,g4,pc):
    w1 = 1/(abs(g1[0]-pc[0])+abs(g1[1]-pc[1]))
    w2 = 1/(abs(g2[0]-pc[0])+abs(g2[1]-pc[1]))
    w3 = 1/(abs(g3[0]-pc[0])+abs(g3[1]-pc[1]))
    w4 = 1/(abs(g4[0]-pc[0])+abs(g4[1]-pc[1]))
    return [w1,w2,w3,w4]


def motion_model(g1,g2,g3,g4,pc,prev_action,allow_motion):
    ad = ar = al = au = 1
    wl = manhattanweights(g1,g2,g3,g4,pc)
    gl = [g1,g2,g3,g4]
    # weighted sum of ghosts position in each direction
    # higher sum, indicates ghosts closer to pacman in respective direction
    for g,w in zip(gl,wl):
        if g[0] > pc[0]:
            ad += w/(g[0] - pc[0])
        elif g[0] < pc[0]:
            au += w/(pc[0] - g[0])
        if g[1] > pc[1]:
            ar += w/(g[1] - pc[1])
        elif g[1] < pc[1]:
            al += w/(pc[1] - g[1])  
    
    # Inverse to obtain higher values indicate more probable direction of motion
    ad = 1/ad
    au = 1/au
    ar = 1/ar
    al = 1/al
    
    # Likely to repeat action, so scale up probability
    if prev_action == 'r':
        ar = ar*1.25
    elif prev_action == 'l':
        al = al*1.25
    elif prev_action == 'u':
        au = au*1.25
    elif prev_action == 'd':
        ad = ad*1.25

    # normalize probabilities
    sum = al + ad + au + ar
    p_l = al/sum
    p_r = ar/sum
    p_u = au/sum
    p_d = ad/sum

    # Given Dictionary of allowable motions, if cannot move direction, split the probabilities to other directions
    if allow_motion['u'] == 0:
        p_r += 0.4*p_u
        p_l += 0.4*p_u
        p_d += 0.2*p_u
        p_u = 0
    
    if allow_motion['d'] == 0:
        p_r += 0.4*p_d
        p_l += 0.4*p_d
        p_u += 0.2*p_u
        p_d = 0
    
    if allow_motion['r'] == 0:
        p_u += 0.4*p_r
        p_d += 0.4*p_r
        p_l += 0.2*p_r
        p_r = 0
    
    if allow_motion['l'] == 0:
        p_u += 0.4*p_l
        p_d += 0.4*p_l
        p_r += 0.2*p_l
        p_l = 0

    return p_l,p_r,p_u,p_d
    







    

        