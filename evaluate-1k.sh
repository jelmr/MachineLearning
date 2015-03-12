python solution.py --output output-nn.csv input/train-1k.csv input/test-1k.csv nn 
python solution.py --output output-rr1.csv input/train-1k.csv input/test-1k.csv rr1 
python solution.py --output output-svr.csv input/train-1k.csv input/test-1k.csv svr
python solution.py --output output-tc.csv input/train-1k.csv input/test-1k.csv tc

echo "nn:" > cprs.txt
python evaluate.py --plot nn.png output-nn.csv input/test-1k.csv >>cprs.txt
echo "rr1:" >> cprs.txt
python evaluate.py --plot rr1.png output-rr1.csv input/test-1k.csv >>cprs.txt
echo "svr:" >> cprs.txt
python evaluate.py --plot svr.png output-svr.csv input/test-1k.csv >>cprs.txt
echo "tc:" >> cprs.txt
python evaluate.py --plot tc.png output-tc.csv input/test-1k.csv >>cprs.txt

cat cprs.txt
