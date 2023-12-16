# Quantization aware training for Neural Networks
Usage <br>

run Final_gcn.ipynb for conducting QAT on GCN<br>
run Final_GPT.ipynb for conducting QAT on GPT <br>
run MobileNet_8bit.ipynb for conducting QAT on MobileNet <br>

# Note:
In Quantization aware training we train scaling factor for both weights and activation. The scaling factor is <br>
basically a clipping value divided by the range specified by the bits. This becomes trainable due to the <br>
fact that STE( straight through estimate ) makes round(x) differentiable . Detail steps are explained in the report <br>
section 3. 
