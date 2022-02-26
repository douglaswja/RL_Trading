import torch
import pandas as pd

class BaseEnvironment:
    
    def __init__(self, data: torch.tensor, bids: pd.Series, asks: pd.Series, use_midprice:bool=False, trading_fee:float=1e-4, neutral_penalty:float=3e-5):
        self.data = data
        self.bids = bids
        self.asks = asks
        self.observation_space = data.shape[-1]
        self.use_midprice = use_midprice
        self.trading_fee = trading_fee
        self.neutral_penalty = neutral_penalty
        
        self.action_space = [0, 1, 2]
        self.action_idx = -1
        self.curr_step = None
    
    def step(self, action: int):
        if curr_step is None:
            raise LookupError("Environment must be reset before it can step forward")
        
        curr_state = self.data[self.curr_step, ...].copy()
        self.curr_step += 1
        
        # next_state
        raw_next_state = self.data[self.curr_step, ...].copy()
        next_state = raw_next_state[..., action_idx] = action
        
        # reward
        reward = get_reward(
            prev_state = curr_state, prev_bid = self.bids[self.curr_step - 1], prev_ask = self.asks[self.curr_step - 1],
            curr_state = next_state, curr_bid = self.bids[self.curr_step], curr_ask = self.asks[self.curr_step],
            use_midprice = self.use_midprice,
            prev_action_idx = self.action_idx,
            trading_fee = self.trading_fee,
            neutral_penalty = self.neutral_penalty,
        )
        
        # is_done
        is_done = self.curr_step == (self.data.shape[0] - 1)
        
        return next_state, reward, is_done
        
    
    def reset(self):
        self.curr_step = 0
        return self.data[self.curr_step, ...]
    
    def close(self):
        self.curr_step = None
    
    
    @staticmethod
    def get_reward(prev_state, prev_bid, prev_ask, curr_state, curr_bid, curr_ask, use_midprice, prev_action_idx, trading_fee, neutral_penalty):
        valid_actions = [0, 1, 2]
        assert (prev_state.dim() == 1) and (curr_state.dim() == 1), "State provided is not 1 dimensional"
        assert (prev_state[prev_action_idx] in valid_actions) and (curr_state[prev_action_idx] in valid_actions), "Action provided is not in [0, 1, 2] range"

        prev_is_neutral = prev_state[prev_action_idx] == 1
        curr_is_neutral = curr_state[prev_action_idx] == 1

        spread = 0 if use_midprice else (prev_ask - prev_bid)

        if curr_is_neutral:
            if prev_is_neutral:
                # Neutral -> Neutral
                return -neutral_penalty

            if not prev_is_neutral:
                # Active -> Neutral
                return -(trading_fee)

        # Here onwards current portfolio is active
        curr_is_long = curr_state[prev_action_idx] == 2 # True -> Long, False -> Short
        if use_midprice and curr_is_long:
            price_change = ((curr_ask + curr_bid) / 2) - ((prev_ask + prev_bid) / 2)
        elif use_midprice and not curr_is_long:
            price_change = ((prev_ask + prev_bid) / 2) - ((curr_ask + curr_bid) / 2)
        elif not use_midprice and curr_is_long:
            price_change = curr_ask - prev_ask
        else:
            price_change = prev_bid - curr_bid

        portfolio_is_same = prev_state[prev_action_idx] == curr_state[prev_action_idx]

        if prev_is_neutral:
            # Neutral -> Active
            return -trading_fee + price_change

        if not prev_is_neutral:
            if portfolio_is_same:
                # Active -> Active
                return price_change
            else:
                # Active -> -Active
                return -(spread + trading_fee) - trading_fee + price_change
