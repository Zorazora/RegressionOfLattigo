package main

import (
	"encoding/csv"
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ckks/bootstrapping"
	"github.com/ldsec/lattigo/v2/rlwe"
	"os"
	"runtime"
	"strconv"
	"time"
)

// Multivariate linear regression
func main() {
	var start time.Time
	x, y :=ReadCsv("housing_price.csv")
	X_train := addBias(x[:5])
	y_train := y[:5]
	//X_train := addBias(x[:250])
	//y_train := y[:250]
	var m int = len(X_train)
	//X_test := addBias(x[5:10])
	//y_test := y[5:10]
	//X_test := addBias(x[250:270])
	//y_test := y[250:270]
	//var n int = len(X_test)
	//fmt.Println(X_train)
	fmt.Println(m)

	fmt.Println("Preprocessing Data... ")
	start = time.Now()
	Avectors, max := fillMatrix(X_train)
	Avectors_T := transpose(Avectors)
	w := make([]complex128, max)
	for i:=0; i<max; i++ {
		w[i] = 0
	}
	AMatDiag := getDiagonal(Avectors, max)
	ATMatDiag := getDiagonal(Avectors_T, max)
	y_train = fillVector(y_train, max)

	rots := make([]int, max)
	for i:=0; i<max; i++ {
		rots[i] = i
	}

	//Testmatrix, max_test := fillMatrix(X_test)
	//rots_test := make([]int, max_test)
	//for i:=0; i<max_test; i++ {
	//	rots_test[i] = i
	//}
	//TestMatDiag := getDiagonal(Testmatrix, max_test)
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Generating Keys... ")
	start = time.Now()
	ckksParams := bootstrapping.DefaultCKKSParameters[0]
	btpParams := bootstrapping.DefaultParameters[0]
	params, _ := ckks.NewParametersFromLiteral(ckksParams)
	kgen := ckks.NewKeyGenerator(params)
	sk, _ := kgen.GenKeyPairSparse(btpParams.H)
	rlk := kgen.GenRelinearizationKey(sk,2)
	encoder := ckks.NewEncoder(params)
	encryptor := ckks.NewEncryptor(params, sk)
	decryptor := ckks.NewDecryptor(params, sk)
	rotkey_train := kgen.GenRotationKeysForRotations(append(rots, 0-max), true, sk)
	//rotkey_test := kgen.GenRotationKeysForRotations(append(rots_test, 0-max_test), true, sk)
	rotations := btpParams.RotationsForBootstrapping(params.LogN(), params.LogSlots())
	rotkey_boot := kgen.GenRotationKeysForRotations(rotations, true, sk)
	bootstrapper, err := bootstrapping.NewBootstrapper(params, btpParams, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkey_boot})
	if err != nil {
		panic(err)
	}
	eval_train := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkey_train})
	//eval_test := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkey_test})
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Encoding... ")
	// Encode w
	start = time.Now()
	ptw := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	ptw = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), w, params.LogSlots())
	ctw := encryptor.EncryptNew(ptw)

	// Encode matrix and matrix transpose
	ctmatrix := make([]*ckks.Ciphertext, max)
	ctmatrix_T := make([]*ckks.Ciphertext, max)
	for i:=0; i<max; i++ {
		ptmatrix := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
		ptmatrix_T := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
		ptmatrix = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), AMatDiag[i], params.LogSlots())
		ptmatrix_T = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), ATMatDiag[i], params.LogSlots())
		ctmatrix[i] = encryptor.EncryptNew(ptmatrix)
		ctmatrix_T[i] = encryptor.EncryptNew(ptmatrix_T)
	}

	// Encode y
	pty := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	pty = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), y_train, params.LogSlots())
	cty := encryptor.EncryptNew(pty)

	// Encode x_test
	//ctmatrix_test := make([]*ckks.Ciphertext, max_test)
	//for i:=0; i<max_test; i++ {
	//	pt := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	//	pt = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), TestMatDiag[i], params.LogSlots())
	//	ctmatrix_test[i] = encryptor.EncryptNew(pt)
	//}

	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Computing... ")
	start = time.Now()

	// fit model
	var alpha float64 = 1
	var epochs int = 8
	//costs := make([]*ckks.Ciphertext, epochs)
	for i:=0; i<epochs; i++  {
		fmt.Println(i)
		y_hat := dot(ctmatrix, ctw, rots, eval_train, params.Scale())
		fmt.Println(y_hat.Scale, cty.Scale)
		fmt.Println(y_hat.Level())
		eval_train.Rescale(y_hat, params.Scale(), y_hat)
		if y_hat.Level()<=2 {
			y_hat = bootstrapper.Bootstrapp(y_hat)
		}
		loss := eval_train.SubNew(y_hat, cty)
		fmt.Println("loss")
		fmt.Println(loss.Scale)
		fmt.Println(encoder.Decode(decryptor.DecryptNew(loss), params.LogSlots())[:m])
		w_grad := eval_train.MultByConstNew(dot(ctmatrix_T, loss, rots, eval_train, params.Scale()), alpha/float64(m))
		fmt.Println("w_grad")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(w_grad), params.LogSlots())[:m])
		fmt.Println("ctw")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(ctw), params.LogSlots())[:m])
		fmt.Println(w_grad.Scale, w_grad.Level(), w_grad.Degree())
		fmt.Println(ctw.Scale, params.Scale())
		eval_train.Rescale(w_grad, params.Scale(), w_grad)
		if w_grad.Level()<=2 {
			w_grad = bootstrapper.Bootstrapp(w_grad)
		}
		ctw = eval_train.SubNew(ctw, w_grad)
		fmt.Println("ctw_new")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(ctw), params.LogSlots())[:m])
	}

	//predict
	//cty_pre := dot(ctmatrix_test, ctw, rots_test, eval_test, params.Scale())
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Decrypting... ")
	start = time.Now()
	fmt.Println("ctw_new")
	fmt.Println(encoder.Decode(decryptor.DecryptNew(ctw), params.LogSlots())[:m])
	fmt.Printf("Done (%s)\n", time.Since(start))
	//y_pre := encoder.Decode(decryptor.DecryptNew(cty_pre), params.LogSlots())[:n]
	//fmt.Printf("Done (%s)\n", time.Since(start))
	//r2 := r2score(y_pre, y_test)
	//fmt.Println(r2)
	//fmt.Println(y_pre)
	//fmt.Println(y_test)
	runtime.GC()
}

func ReadCsv(filename string) (x [][]complex128, y []complex128){
	opencast, err := os.Open(filename)
	if err!=nil {fmt.Println(err)}
	defer opencast.Close()

	ReadCsv := csv.NewReader(opencast)
	ReadAll, err := ReadCsv.ReadAll()
	x = make([][]complex128, len(ReadAll))
	y = make([]complex128, len(ReadAll))
	for i:=0; i<len(ReadAll); i++ {
		value_x0, err := strconv.ParseFloat(ReadAll[i][0], 64)
		if err!=nil {fmt.Println(err)}
		value_x1, err := strconv.ParseFloat(ReadAll[i][1], 64)
		if err!=nil {fmt.Println(err)}
		value_x2, err := strconv.ParseFloat(ReadAll[i][2], 64)
		if err!=nil {fmt.Println(err)}
		value_y, err := strconv.ParseFloat(ReadAll[i][3], 64)
		if err!=nil {fmt.Println(err)}
		x[i] = make([]complex128, 3)
		x[i][0] = complex(value_x0, 0)
		x[i][1] = complex(value_x1, 0)
		x[i][2] = complex(value_x2, 0)
		y[i] = complex(value_y, 0)
	}
	return x, y
}

func addBias(x [][]complex128) (x_add [][]complex128){
	x_add = make([][]complex128, len(x))
	for i:=0; i<len(x); i++ {
		x_add[i] = make([]complex128, 1)
		x_add[i][0] = complex(1,0)
		x_add[i] = append(x_add[i], x[i]...)
	}
	return x_add
}

func fillMatrix(x [][]complex128) (Avectors [][]complex128, max int) {
	if len(x)>len(x[0]){
		max = len(x)
	}else {
		max = len(x[0])
	}
	Avectors = make([][]complex128, max)
	for i := range Avectors {
		tmp := make([]complex128, max)
		for j := 0; j < len(x[0]); j++ {
			tmp[j] = x[i][j]
		}
		Avectors[i] = tmp
	}
	return Avectors, max
}

func fillVector(y []complex128, max int) (hvector []complex128) {
	hvector = make([]complex128, max)
	for i:= range y {
		hvector[i] = y[i]
	}
	return hvector
}

func getDiagonal(x [][]complex128, max int) (diagMatrix map[int][]complex128) {
	diagMatrix = make(map[int][]complex128)
	for i := 0; i < max; i++ {
		tmp := make([]complex128, max)
		for j := 0; j < max; j++ {
			tmp[j] = x[j%max][(j+i)%max]
		}
		diagMatrix[i] = tmp
	}
	return diagMatrix
}

func dot(X []*ckks.Ciphertext, v *ckks.Ciphertext, rots []int, eval ckks.Evaluator, scale float64) (res *ckks.Ciphertext){
	v_new := eval.RotateNew(v, 0-len(rots))
	v_dup := eval.AddNew(v, v_new)
	ctrots := eval.RotateHoistedNew(v_dup, rots)
	res = eval.MulRelinNew(X[0], ctrots[0])
	//eval.Rescale(res, scale, res)
	for i:=1; i<len(X); i++ {
		eval.Add(res, eval.MulRelinNew(X[i], ctrots[i]), res)
		//eval.Rescale(res, scale, res)
	}
	//eval.Relinearize(res, res)
	return res
}

func transpose(X [][]complex128) (X_T [][]complex128) {
	X_T = make([][]complex128, len(X[0]))
	for i:=0; i<len(X[0]); i++ {
		X_T[i] = make([]complex128, len(X))
		for j:=0; j<len(X); j++ {
			X_T[i][j] = X[j][i]
		}
	}
	return X_T
}

func r2score(y_pred, y []complex128) (r2 float64) {
	rss := float64(0)
	y_sum := float64(0)
	for i:=0; i<len(y); i++ {
		rss += (real(y_pred[i])-real(y[i]))*(real(y_pred[i])-real(y[i]))
		y_sum += real(y[i])
	}
	y_mean := y_sum/float64(len(y))
	tss := float64(0)
	for i:=0; i<len(y); i++ {
		tss += (real(y[i])-y_mean)*(real(y[i])-y_mean)
	}
	r2 = 1 - (rss/tss)
	return r2
}