## Skriver kostnadsfunktionerna här ##

def Func_price_from_capa(cap):
    return 5

def Func_el_consum(soc, cap):
    return 5

def Func_time_charge(soc, cap):
    return 5

""" Ska inte vara här, enbart för testningmöjligheter"""
def get_avail_value(avail: list, state: list):
    tot_char_avail = 0
    for i in range(len(avail)):
        tot_char_avail += avail[i]*state[i]
    print(state[0])
    print(avail)
    proc_avail = tot_char_avail/state[0]
    return proc_avail, tot_char_avail
    
    