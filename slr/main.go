package main

import (
	"encoding/csv"
	"fmt"
	"github.com/ldsec/lattigo/v2/ckks"
	"github.com/ldsec/lattigo/v2/ckks/bootstrapping"
	"github.com/ldsec/lattigo/v2/ring"
	"github.com/ldsec/lattigo/v2/rlwe"
	"github.com/ldsec/lattigo/v2/utils"
	"os"
	"runtime"
	"strconv"
	"time"
)

// simple linear regression

type Context struct {
	params      ckks.Parameters
	ringQ      *ring.Ring
	ringP      *ring.Ring
	ringQP     *ring.Ring
	prng        utils.PRNG
	encoder     ckks.Encoder
	kgen        rlwe.KeyGenerator
	sk         *rlwe.SecretKey
	pk         *rlwe.PublicKey
	rlk        *rlwe.RelinearizationKey
	encryptorPk ckks.Encryptor
	encryptorSk ckks.Encryptor
	decryptor   ckks.Decryptor
	evaluator   ckks.Evaluator
}

func main() {
	var start time.Time
	//x := []complex128{67, 21, 20, 36, 15, 62, 85, 4, 51, 3}
	//y := []complex128{78, 33, 36, 52, 20, 65, 83, 12, 57, 16}
	x, y := ReadCsv("slr-train.csv")
	x = x[:40]
	y = y[:40]
	var len int = len(x)

	rots := make([]int, len)
	for i:=0; i<len; i++ {
		rots[i] = 0-i
	}

	// Generate keys params
	fmt.Println("Generate key time")
	start = time.Now()
	ckksParams := bootstrapping.DefaultCKKSParameters[0]
	btpParams := bootstrapping.DefaultParameters[0]
	params, err := ckks.NewParametersFromLiteral(ckksParams)
	if err != nil {
		panic(err)
	}
	ctx := new(Context)
	ctx.params = params
	ctx.kgen = ckks.NewKeyGenerator(ctx.params)
	ctx.sk, ctx.pk = ctx.kgen.GenKeyPairSparse(btpParams.H)
	if ctx.prng, err = utils.NewPRNG(); err != nil {
		panic(err)
	}
	ctx.rlk = ctx.kgen.GenRelinearizationKey(ctx.sk, 2)
	ctx.encoder = ckks.NewEncoder(params)
	ctx.encryptorPk = ckks.NewEncryptor(params, ctx.pk)
	ctx.encryptorSk = ckks.NewEncryptor(params, ctx.sk)
	ctx.decryptor = ckks.NewDecryptor(params, ctx.sk)
	ctx.evaluator = ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: ctx.rlk})
	rotkey_innersum := ctx.kgen.GenRotationKeysForRotations(ctx.params.RotationsForInnerSum(1,len), false, ctx.sk)
	ctx.evaluator = ctx.evaluator.WithKey(rlwe.EvaluationKey{Rlk: ctx.rlk, Rtks: rotkey_innersum})
	rotkey := ctx.kgen.GenRotationKeysForRotations(rots, true, ctx.sk)
	eval := ckks.NewEvaluator(params, rlwe.EvaluationKey{Rlk: ctx.rlk, Rtks: rotkey})
	fmt.Printf("Done in %s \n", time.Since(start))

	// Encrypt
	fmt.Println("Encrypt time")
	start = time.Now()
	x_plaintext := ckks.NewPlaintext(ctx.params, ctx.params.MaxLevel(), ctx.params.Scale())
	x_plaintext = ctx.encoder.EncodeNTTAtLvlNew(ctx.params.MaxLevel(),x, ctx.params.LogSlots())
	x_ciphertext := ctx.encryptorPk.EncryptNew(x_plaintext)

	y_plaintext := ckks.NewPlaintext(ctx.params, ctx.params.MaxLevel(), ctx.params.Scale())
	y_plaintext = ctx.encoder.EncodeNTTAtLvlNew(ctx.params.MaxLevel(), y, ctx.params.LogSlots())
	y_ciphertext := ctx.encryptorPk.EncryptNew(y_plaintext)

	pthelper := ckks.NewPlaintext(ctx.params, ctx.params.MaxLevel(), ctx.params.Scale())
	pthelper = ctx.encoder.EncodeNTTAtLvlNew(ctx.params.MaxLevel(), []complex128{complex(1,0)}, ctx.params.LogSlots())
	cthelper := ctx.encryptorPk.EncryptNew(pthelper)
	fmt.Printf("Done in %s \n", time.Since(start))

	// Compute
	fmt.Println("Compute time")
	start = time.Now()
	meanx := Mean(x_ciphertext, ctx.evaluator, len)
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meanx), ctx.params.LogSlots())[0])
	ctx.evaluator.MulRelin(meanx, cthelper, meanx)
	meany := Mean(y_ciphertext, ctx.evaluator, len)
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meany), ctx.params.LogSlots())[0])
	ctx.evaluator.MulRelin(meany, cthelper, meany)

	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meanx), ctx.params.LogSlots())[0])
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meany), ctx.params.LogSlots())[0])

	meanx_list := format(meanx, rots, len, eval, ctx)
	meany_list := format(meany, rots, len, eval, ctx)

	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meanx_list), ctx.params.LogSlots())[:len])
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(meany_list), ctx.params.LogSlots())[:len])

	covariance := Covariance(x_ciphertext, y_ciphertext, meanx_list,
		meany_list, len, ctx.evaluator,ctx)
	variancex := Covariance(x_ciphertext, x_ciphertext, meanx_list,
		meany_list, len, ctx.evaluator,ctx)


	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(covariance), ctx.params.LogSlots())[0])
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(variancex), ctx.params.LogSlots())[0])

	variancex_decrypt := real(ctx.encoder.Decode(ctx.decryptor.DecryptNew(variancex), ctx.params.LogSlots())[0])
	constant := 1/float64(variancex_decrypt)
	m := ctx.evaluator.MultByConstNew(covariance, constant)
	//fmt.Println(ctx.encoder.Decode(ctx.decryptor.DecryptNew(m), ctx.params.LogSlots())[0])

	m_mul_meanx := ctx.evaluator.MulRelinNew(meanx, m)
	ctx.evaluator.Rescale(m_mul_meanx, ctx.params.Scale(), m_mul_meanx)
	ctx.evaluator.Rescale(meany, ctx.params.Scale(), meany)
	c := ctx.evaluator.SubNew(meany, m_mul_meanx)

	fmt.Printf("Done in %s \n", time.Since(start))

	// Decrypt
	fmt.Println("Decrypt time")
	start = time.Now()
	m_decrypt := ctx.encoder.Decode(ctx.decryptor.DecryptNew(m), ctx.params.LogSlots())[0]
	c_decrypt := ctx.encoder.Decode(ctx.decryptor.DecryptNew(c), ctx.params.LogSlots())[0]
	fmt.Println("m: ", real(m_decrypt))
	fmt.Println("c: ", real(c_decrypt))
	fmt.Printf("Done in %s \n", time.Since(start))
	runtime.GC()
}

func Mean(ciphertext *ckks.Ciphertext, eval ckks.Evaluator, len int) (res *ckks.Ciphertext)  {
	res = ciphertext.CopyNew()
	eval.InnerSum(ciphertext, 1, len, res)
	constant := 1/float64(len)
	eval.MultByConst(res, constant, res)
	return res
}

func Covariance(x, y *ckks.Ciphertext, meanx,meany *ckks.Ciphertext, len int, eval ckks.Evaluator, ctx *Context) (res *ckks.Ciphertext) {
	eval.Rescale(meanx, ctx.params.Scale(), meanx)
	subx := eval.SubNew(x, meanx)
	eval.Rescale(meany, ctx.params.Scale(), meany)
	suby := eval.SubNew(y, meany)
	mul := eval.MulRelinNew(subx, suby)
	res = mul.CopyNew()
	eval.InnerSum(mul, 1, len, res)
	return res
}

func format(x *ckks.Ciphertext, rots []int, len int, eval ckks.Evaluator, ctx *Context) (res *ckks.Ciphertext) {
	ctrots := eval.RotateHoistedNew(x, rots)
	res = ctrots[0].CopyNew()
	for i:=1; i<len; i++ {
		res = eval.AddNew(res, ctrots[0-i])
	}
	return res
}

func ReadCsv(filename string)(x []complex128, y []complex128) {
	opencast, err := os.Open(filename)
	if err!=nil {fmt.Println(err)}
	defer opencast.Close()

	ReadCsv := csv.NewReader(opencast)
	ReadAll, err := ReadCsv.ReadAll()
	x = make([]complex128, len(ReadAll))
	y = make([]complex128, len(ReadAll))
	for i:=0; i<len(ReadAll); i++ {
		value_x, err := strconv.ParseFloat(ReadAll[i][0], 64)
		if err!=nil {fmt.Println(err)}
		value_y, err := strconv.ParseFloat(ReadAll[i][1], 64)
		if err!=nil {fmt.Println(err)}
		x[i] = complex(value_x, 0)
		y[i] = complex(value_y, 0)
	}
	return x, y
}