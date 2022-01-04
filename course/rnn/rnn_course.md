# RNN (Recurrent Neural Network) course

**h0 = 0** most of the time

- **Vanilla RNN** : RNN simple, y^ = f(h,x)

- **LSTM** : ^y = flstm(h,x)

- **GRU** (Gated Recurrent Unit), ^y = fgru(h,x)

- Classification : dernier y^ dans les couches denses suivantes (apres les couches recurrentes)
- Regression : y^ intermediaires dans les couches denses suivantes (apres les couches recurrentes)