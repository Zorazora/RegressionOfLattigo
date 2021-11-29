package main

import (
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ckks/bootstrapping"
	"github.com/ldsec/lattigo/v2/rlwe"
	"math/cmplx"
	"time"
)

func main() {
	var start time.Time
	x := [][]complex128{{31.3},{29.7},{31.3},{31.8},{31.4},{31.9},{31.8},{31.0},{29.7},{31.4},{32.4},{33.6},{30.2},{30.4},{27.6},{31.8},{31.3},{34.5},{28.9},{28.2}}
	y := []complex128{1,1,0,0,1,1,1,1,0,1,1,1,0,0,0,1,1,1,0,0}
	X_train := addBias(x[:5])
	y_train := y[:5]
	var m int = len(X_train)
	//X_test := addBias([][]complex128{{32.2},{29.8},{33.1}})
	//y_test := []complex128{1,0,1}
	//var n int = len(X_test)

	fmt.Println("Preprocessing the Data")
	start = time.Now()
	Avectors, max := fillMatrix(X_train)
	Avectors_T := transpose(Avectors)
	theta := make([]complex128, max)
	for i:=0; i<max; i++ {
		theta[i] = 0
	}

	AMatDiag := getDiagonal(Avectors, max)
	ATMatDiag := getDiagonal(Avectors_T, max)
	y_train = fillVector(y_train, max)

	rots := make([]int, max)
	for i:=0; i<max; i++ {
		rots[i] = i
	}

	helper := make([]complex128, max)
	for i:=0; i<max; i++ {
		helper[i] = complex(1,0)
	}
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
	eval_train := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: rlk, Rtks: rotkey_train})
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Encoding... ")
	// Encode theta
	start = time.Now()
	ptw := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	ptw = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), theta, params.LogSlots())
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

	pthelper := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	pthelper = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), helper, params.LogSlots())
	cthelper := encryptor.EncryptNew(pthelper)

	// Encode y
	pty := ckks.NewPlaintext(params, params.MaxLevel(), params.Scale())
	pty = encoder.EncodeNTTAtLvlNew(params.MaxLevel(), y_train, params.LogSlots())
	cty := encryptor.EncryptNew(pty)
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Computing... ")
	start = time.Now()

	var alpha float64 = 0.001
	var epochs int = 2

	for i:=0; i<epochs; i++ {
		tmp := dot(ctmatrix, ctw, rots, eval_train, params.Scale(), encoder, decryptor, params)
		fmt.Println("tmp")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(tmp), params.LogSlots())[:2*m])
		h := sigmoid(tmp, eval_train, -8, 8, 10, params)
		fmt.Println("h")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(h), params.LogSlots())[:2*m])
		diff := eval_train.SubNew(h, cty)
		fmt.Println("diff")
		eval_train.MulRelin(diff, cthelper, diff)
		fmt.Println(encoder.Decode(decryptor.DecryptNew(diff), params.LogSlots())[:2*m])
		tmp = dot(ctmatrix_T, diff, rots, eval_train, params.Scale(), encoder, decryptor, params)
		fmt.Println("tmp")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(tmp), params.LogSlots())[:2*m])
		grad := eval_train.MultByConstNew(tmp, alpha/float64(m))
		eval_train.Rescale(grad, params.Scale(), grad)
		fmt.Println("grad")
		fmt.Println(encoder.Decode(decryptor.DecryptNew(grad), params.LogSlots())[:2*m])
		eval_train.Sub(ctw, grad, ctw)
		fmt.Println("ctw")
		fmt.Println(ctw.Level(), ctw.Degree())
		fmt.Println(encoder.Decode(decryptor.DecryptNew(ctw), params.LogSlots())[:2*m])
	}
	fmt.Printf("Done (%s)\n", time.Since(start))

	fmt.Println("Decrypting... ")
	start = time.Now()
	fmt.Println(encoder.Decode(decryptor.DecryptNew(ctw), params.LogSlots())[:2*m])
	fmt.Printf("Done (%s)\n", time.Since(start))
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
		fmt.Println(diagMatrix[i])
	}
	return diagMatrix
}

func dot(X []*ckks.Ciphertext, v *ckks.Ciphertext, rots []int, eval ckks.Evaluator, scale float64,
	encoder ckks.Encoder, decryptor ckks.Decryptor, params ckks.Parameters) (res *ckks.Ciphertext){
	v_new := eval.RotateNew(v, 0-len(rots))
	//fmt.Println("v_new")
	//fmt.Println(encoder.Decode(decryptor.DecryptNew(v_new), params.LogSlots())[:2*len(X)])
	v_dup := eval.AddNew(v, v_new)
	//fmt.Println("v_dup")
	//fmt.Println(encoder.Decode(decryptor.DecryptNew(v_dup), params.LogSlots())[:2*len(X)])
	ctrots := eval.RotateHoistedNew(v_dup, rots)
	//fmt.Println("ctrots")
	//for i:=0; i<len(X); i++ {
	//	fmt.Println(encoder.Decode(decryptor.DecryptNew(ctrots[i]), params.LogSlots())[:2*len(X)])
	//}
	//fmt.Println("Finished")
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

func sigmoid(x *ckks.Ciphertext, eval ckks.Evaluator, a,b complex128, degree int, params ckks.Parameters) (res *ckks.Ciphertext)  {
	res = x.CopyNew()
	chebyapproximation := ckks.Approximate(f, a, b, degree)
	a = chebyapproximation.A
	b = chebyapproximation.B
	eval.MultByConst(res, 2/(b-a), res)
	eval.AddConst(res, (-a-b)/(b-a), res)
	eval.Rescale(res, params.Scale(), res)
	res, err := eval.EvaluatePoly(res, chebyapproximation, res.Scale)
	if err!=nil {
		panic(err)
	}
	return res
}

func f(x complex128) complex128 {
	return 1 / (cmplx.Exp(-x) + 1)
}
