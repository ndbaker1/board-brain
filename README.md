# c4spinner

An ML experiment on Connect 4 Spin

## Example

```
❯ cargo r --release
    Finished release [optimized] target(s) in 0.10s
     Running `target/release/c4spin`
┌  🧠 Board-Brain
│
◇  Select game
│  connect-4-spin
│
◇  ✅ components loaded
│
◇  Select model
│  Multi-Layer Perceptron
│
●  📦 loading existing model parameters..
│
├  🧠 training model..
│
Step:   150 | Loss:  0.02589 | LR: 0.10000
Step:   300 | Loss:  0.05494 | LR: 0.10000
Step:   450 | Loss:  0.00062 | LR: 0.10000
Step:   600 | Loss:  0.00663 | LR: 0.10000
Step:   750 | Loss:  0.01912 | LR: 0.10000
Step:   900 | Loss:  0.02112 | LR: 0.10000
Step:  1050 | Loss:  0.05442 | LR: 0.10000
Step:  1200 | Loss:  0.05152 | LR: 0.10000
Step:  1350 | Loss:  0.02225 | LR: 0.10000
Step:  1500 | Loss:  0.01937 | LR: 0.10000
Step:  1650 | Loss:  0.09836 | LR: 0.10000
Step:  1800 | Loss:  0.02883 | LR: 0.10000
Step:  1950 | Loss:  0.01182 | LR: 0.10000
Step:  2100 | Loss:  0.06474 | LR: 0.10000
Step:  2250 | Loss:  0.00020 | LR: 0.10000
Step:  2400 | Loss:  0.00864 | LR: 0.10000
Step:  2550 | Loss:  0.00280 | LR: 0.10000
Step:  2700 | Loss:  0.09412 | LR: 0.10000
Step:  2850 | Loss:  0.10084 | LR: 0.10000
Step:  3000 | Loss:  0.02429 | LR: 0.10000
Step:  3150 | Loss:  0.01370 | LR: 0.10000
Step:  3300 | Loss:  0.01920 | LR: 0.10000
Step:  3450 | Loss:  0.02685 | LR: 0.10000
Step:  3600 | Loss:  0.02566 | LR: 0.10000
Step:  3750 | Loss:  0.01335 | LR: 0.10000
Step:  3900 | Loss:  0.02021 | LR: 0.10000
◆  💾 saving model..
│
└  ✅ model ready

🎮 creating game with model..
1🟩🟩🟩🟩🟩
2🟩🟩🟩🟩🟩
3🟩🟩🟩🟩🟩
4🔵🟩🟩🟩🟩
5🟩🟩🟩🟩🟩
6🟩🟩🟩🟩🟩
7🟩🟩🟩🟩🟩
8🟩🟩🟩🟩🟩
 a b c d e
where would you like to go [ex. d3]>
```
