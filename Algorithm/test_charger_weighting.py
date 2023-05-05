# _1*x + _2*x + _3*(2x) + ... = 1

def weight_avail(avail):
    tot_xs = avail[0]
    for i in range(1, len(avail)):
        tot_xs += avail[i] * i

    x_val = 1/tot_xs

    weighted_avail = avail.copy()
    weighted_avail[0] = avail[0] * x_val
    for i in range(1, len(avail)):
        weighted_avail[i] = avail[i] * (x_val * i)

    return weighted_avail

if __name__ == "__main__":
    print(1-weight_avail([0.5, 0.5])[0]) 
    print(1-weight_avail([0.5, 0.0, 0.5])[0])
    print(1-weight_avail([0.9, 0.05, 0, 0, 0, 0, 0, 0, 0.05])[0])
    print(1-weight_avail([0.9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1])[0])

    #Big test
    a = [0]*10_000
    a[0] = 0.99
    a[-1] = 0.01
    print(1-weight_avail(a)[0])

    charge_cost_50kw = 100
    charge_cost_100kw = 50

    charge_cost_50kw = charge_cost_50kw / (1-weight_avail(a)[0])
    charge_cost_100kw = charge_cost_100kw / (1-weight_avail([0.5, 0.5])[0])

    print(charge_cost_50kw)
    print(charge_cost_100kw)

