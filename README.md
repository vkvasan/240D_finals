# Quantization aware training for Neural Networks
Usage 

run Final_gcn.ipynb for conducting QAT on GCN
run Final_GPT.ipynb for conducting QAT on GPT 
run MobileNet_8bit.ipynb for conducting QAT on MobileNet 

# Note:
In Quantization aware training we train scaling factor for both weights and activation. The scaling factor is
basically a clipping value divided by the range specified by the bits. This becomes trainable due to the 
fact that STE( straight through estimate ) makes round(x) differentiable . Detail steps are explained in the report 
section 3. 
