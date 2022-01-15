import pandas as pd

def load_message(filename):
    """
    filename: csv file
    
        Event Type:
        1: Submission of a new limit order
        2: Cancellation (partial deletion of a limit order)
        3: Deletion (total deletion of a limit order)
        4: Execution of a visible limit order
        5. direction:  
            +1: indicats: BUY order // seller initiated trade (a seller takes the initiative to consume some of the quoted offers at the BID// The volume at the BID decreases 
            -1: indicates: SELL order // buyer initiated trade (a buyer takes the initiative to consume some of the quoted offers at the ASK// The volume at the ASK decreases 
        6: Indicates a cross trade, e.g. auction trade
        7: Trading halt indicator (detailed information below)
    """
    message = pd.read_csv(filename, header=None, low_memory=False)
    message = message.drop(columns=[6])
    message = message.rename(columns={0:"time", 1:"type", 2: "id", 3: "vol", 4: "price", 5:"direct"})
    return message

def load_LOB(filename):
    """
    filename: csv file
    """
    #load data
    LOB = pd.read_csv(filename, header=None, low_memory=False)
    
    #rename columns
    kont = 0
    level = 1
    dico = {}
    n_levels = len(LOB.columns)//4
    for _ in range(n_levels):
        dico[kont] = f"ask_price_{level}"
        kont += 1
        dico[kont] = f"ask_size_{level}"
        kont += 1
        dico[kont] = f"bid_price_{level}"
        kont += 1
        dico[kont] = f"bid_size_{level}"
        kont += 1
        level += 1
    LOB = LOB.rename(columns=dico)
    return LOB