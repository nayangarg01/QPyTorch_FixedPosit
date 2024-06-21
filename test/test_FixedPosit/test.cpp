#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#define FP16_LIMB_SIZE 16
#define FP16_TYPE uint16_t
#define FP32_TYPE uint32_t


uint32_t int32_constants[16];

uint64_t int64_constants[2];

#define SIGN_MASK 0x8000                     // 8000 => 1000 0000 0000 0000(16 bits)
#define FLOAT_SIGN_MASK 0x80000000           // => 1000 0000 0000 0000 0000 0000 0000 0000 (32 bits)
#define FLOAT_SIGN_RESET_MASK 0x7FFFFFFF     // => 1 0 bit 31 1 bits
#define SECOND_BIT_MASK 0x4000               // 4000 => 0100 0000 0000 0000
#define FIXED_POSIT_INF 0x0000               // defining the inf for posit
#define FIXED_POSIT_LIMB_ALL_BITS_SET 0xffff // all bits set for a posit
#define SINGLE_PRECISION_BIAS 127            // bias for the single precision number
#define FLOAT_SIZE 32                        // single precision
#define FLOAT_EXPONENT_MASK 0x7f800000       //=> 0111 1111 1000 0000 0000 0000 0000 0000
#define FLOAT_FRACTION_MASK 0x007fffff       //=> 0000 0000 0111 1111 1111 1111 1111 1111
#define FLOAT_SIGN_SHIFT 31
#define FLOAT_EXPONENT_SHIFT 23
#define FLOAT_DENORMAL_EXPONENT -126
#define FLOAT_HIDDEN_BIT_SET_MASK 0x00800000   //=> 0000 0000 1000 0000 0000 0000 0000 0000
#define FLOAT_SIGN_PLUS_EXP_LENGTH_MINUS_ONE 8 // name defined
#define TEMP_TYPE uint64_t                     // defining the temporary type to be unint64_t
#define UNSIGNED_LONG_LONG_SIZE 64             // name defined
#define EDP_ACC_SIZE 63                        // setting the accumulator size
#define FIXED_POSIT_EXP_SHIFT 41               // 64-23
#define FLOAT_EXP_SIGN_SHIFT 30
#define FLOAT_INF 0x7F800000         // 0111 1111 1000 0000 0000 0000 0000 0000
#define FLOAT_SIGN_PLUS_EXP_LENGTH 9 // name defined
#define FIXED_POSIT_LENGTH_PLUS_ONE 17 /

#define GET_MAX(a, b)           \
    ({                          \
        __typeof__(a) _a = (a); \
        __typeof__(b) _b = (b); \
        _a > _b ? _a : _b;      \
    })

#define _G_INFP 32768

union Bits
{
    float f;
    int32_t si;
    uint32_t ui;
};

typedef FP16_TYPE fp16;
typedef FP32_TYPE fp32;
#define _G_FPOSIT_SHIFT_AMOUNT int32_constants[0]
#define _G_MAXREALFP int32_constants[1]
#define _G_MINREALFP int32_constants[2]
#define FPOSIT_EXTRA_BITS_SHIFT int32_constants[3]
#define _G_USEED int32_constants[4]
#define _G_USEED_ZEROS int32_constants[5]
#define FPOSIT_EXPONENT_MASK int32_constants[6]

#define _G_MAXREAL_INT int32_constants[7]
#define _G_MINREAL_INT int32_constants[8]
#define _G_NBITS int32_constants[9]
#define _G_ESIZE int32_constants[10]

#define _G_RSIZE int32_constants[11]
#define FPOSIT_REGIME_MASK int32_constants[12]
#define FPOSIT_FRACTION_MASK int32_constants[13]
#define _G_FSIZE int32_constants[14]
#define _FP_REGIME_BIAS int32_constants[15]
// define regime size

#define FPOSIT_EXTRA_BITS_MASK int64_constants[0]
#define FPOSIT_HALFWAY_BIT_MASK int64_constants[1]

void generate_fixedposit_constants(int nsize, int es, int rf, uint32_t *int32_constants, uint64_t *int64_constants)
{
    // local vars have the same name as global constant vars, confusing but less likely error can happen here.
    // ugly but it's the straightforward conversion from the original #define macroes;
    // todo: make this one less messy

    _G_NBITS = nsize;
    _G_ESIZE = es;
    _G_RSIZE = rf;
    _G_FSIZE = nsize - es - rf - 1;

    if (nsize <= 16 && rf <= 4)
    {
        _G_FPOSIT_SHIFT_AMOUNT = FP16_LIMB_SIZE - nsize;
        // _G_MAXREALFP = ((1 << (nsize - 1)) - 1) << _G_FPOSIT_SHIFT_AMOUNT;
        _G_MAXREALFP = (1<<(_G_ESIZE+_G_RSIZE)-1)<<(_G_FSIZE+_G_FPOSIT_SHIFT_AMOUNT);
        // _G_MINREALFP = 1<<_G_FPOSIT_SHIFT_AMOUNT;
        _G_MINREALFP = 0<<_G_FPOSIT_SHIFT_AMOUNT;
        FPOSIT_EXTRA_BITS_SHIFT = UNSIGNED_LONG_LONG_SIZE - nsize + 1;
        _G_USEED = 1 << (1 << es);
        _G_USEED_ZEROS = (1 << es);
        FPOSIT_EXPONENT_MASK = _G_USEED_ZEROS - 1;
        _FP_REGIME_BIAS = (1 << (rf - 1)) - 1;

        _G_MAXREAL_INT = (((((1 << rf) - 1 - _FP_REGIME_BIAS) * (_G_USEED_ZEROS)) + ((1 << es) - 1))+SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // _G_MAXREAL_INT = ((_G_USEED_ZEROS * (nsize - 2)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // this is the maximum float integer that can be represented by the fixed posit.
        _G_MINREAL_INT = ((-1* (_FP_REGIME_BIAS) * (_G_USEED_ZEROS))+SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // _G_MINREAL_INT = ((_G_USEED_ZEROS * (2 - nsize)) + SINGLE_PRECISION_BIAS) << FLOAT_EXPONENT_SHIFT;
        // this is the minimum float integer that can be represented by the fixed posit.
        FPOSIT_REGIME_MASK = ((1 << rf) - 1) << (_G_FSIZE + _G_ESIZE);
        FPOSIT_FRACTION_MASK = ((1 << _G_FSIZE) - 1);

        FPOSIT_EXTRA_BITS_MASK = (1UL << (UNSIGNED_LONG_LONG_SIZE - nsize)) - 1;
        FPOSIT_HALFWAY_BIT_MASK = 1UL << (UNSIGNED_LONG_LONG_SIZE - nsize);
    }
    else
    {
        printf("unexpected fposit config\n");
        exit(1);
    }
};

float fixedp16tofp32(fp16 Fp, uint32_t *int32_constants, uint64_t *int64_constants)
{
    union Bits v;
    union Bits v2;
    if(Fp == 32768){
    
    v.si ^= (0 ^ v.si) & -(Fp == 32768);               // if we had 0 in posit, we get 0 in float
    return v.f;
    }
    v2.ui = Fp;
    // printf("the fraction is: %d \n", Fp);
    bool sign = Fp & SIGN_MASK;
    // printf("the sign is: %d \n", sign);
    Fp = (Fp ^ -sign) + sign; // taking 2s complement if sign is 1, else keeping same
    // printf("the fraction is: %d \n", Fp);
    // int  = _G_NBITS - _G_ESIZE - 1
    Fp = Fp>>_G_FPOSIT_SHIFT_AMOUNT;
    v.ui = ((Fp & FPOSIT_REGIME_MASK) >> (_G_ESIZE + _G_FSIZE)) - _FP_REGIME_BIAS; // for regime
    // printf("the current regime is: %d \n", v.ui);
    v.ui = v.ui << _G_ESIZE;
    // printf("the current regime is: %d \n", v.ui);
    // printf("the current float formed is: %f \n", v.f);
    uint32_t exp = (((Fp >> _G_FSIZE) & FPOSIT_EXPONENT_MASK));
    // printf("the current exponent is: %d \n", exp);
    v.ui = (v.ui | exp) + SINGLE_PRECISION_BIAS; // included regime and exponent in the float.
    v.ui = v.ui << 23;
    // printf("the current float formed is: %f \n", v.f);
    uint32_t frac = (Fp & FPOSIT_FRACTION_MASK);
    frac = frac << (23 - _G_FSIZE);
    v.ui = v.ui | frac; // included fraction in the float.
    // printf("the current float formed is: %f \n", v.f);
    v.si ^= (FLOAT_INF ^ v.si) & -(v2.ui == _G_INFP); // if we had inf in posit, we get inf in float
    

    v.ui |= (sign << FLOAT_SIGN_SHIFT); // putting the sign bit in front
    // printf("the current float formed is: %f \n", v.f);
    return v.f;
}

fp16 fp32tofixedp16(float f, uint32_t *int32_constants, uint64_t *int64_constants)
{
    fp16 Fp = 0; // initiallising a 16 bit space for the posit
    union Bits v; // initiallising the union bit space to decode single precision float
    union Bits v2;
    v.f = f; // assigning the bit pattern to the union space
    v2.f = f;
    bool sign = (v.ui & FLOAT_SIGN_MASK); // extracting the sign of the float
    
    v.ui &= 0x7FFFFFFF; // removing the sign from the union space

#ifdef FLOAT_ROUNDING
    uint16_t roundSign = sign << 15;
    if (v.ui > _G_MAXREAL_INT)
        return _G_INFP | roundSign;
    if (v.ui < _G_MINREAL_INT)
        return 0;
#endif
    
    // printf("float is : %d \n",v.ui);
    // min Fposit exFponent in 16, 3 is -112
    // therefore all the float subnormals will be handled
    // in the Fprevious if statement
    // printf("the current output value is: %d \n", Fp);
    // get exponent sign
    bool exp_sign = !(((v.ui&FLOAT_EXPONENT_MASK)>>23)>=SINGLE_PRECISION_BIAS);
    // bool exp_sign = !(v.ui>> FLOAT_EXP_SIGN_SHIFT); // getting the sign of the exp, since it has a bias, 1 means + and 0 means -. hence the !
    // printf("the exp_sign value is: %d \n", exp_sign);
    // get regime and exponent
    uint32_t exp = abs((v.si >> FLOAT_EXPONENT_SHIFT) - SINGLE_PRECISION_BIAS); // getting absolute value of the exponent of the float
    // printf("the exponent value is: %d \n", exp);
    uint32_t regime_and_exp = ((((((exp >> _G_ESIZE) + _FP_REGIME_BIAS))) << (_G_ESIZE)) | (exp & FPOSIT_EXPONENT_MASK)) << (_G_FSIZE);
    // printf("the regime and exp value is: %d \n", regime_and_exp);
    // if exponent is negative
    // regime_and_exp = ((regime_and_exp ^ -exp_sign) + exp_sign);// >> ((exp_sign & !((exp & FPOSIT_EXPONENT_MASK))) & (bool)exp);
    if(exp_sign){
    regime_and_exp = FPOSIT_REGIME_MASK-regime_and_exp-(1<<(_G_FSIZE+_G_ESIZE));
    // printf("the regime and exp value is: %d \n", regime_and_exp);
    }
    // Fp =  (v.si >= _G_MAXREAL_INT);
    // regime_and_exp &= (FPOSIT_REGIME_MASK+(FPOSIT_EXPONENT_MASK<<_G_FSIZE));
    // printf("the regime and exp value is: %d \n", regime_and_exp);
    // printf("the current output value is: %d \n", v.ui);
    if((v.si >= _G_MAXREAL_INT)){
      
      Fp ^= (Fp ^ _G_MAXREALFP) & -(v.si >= _G_MAXREAL_INT);              // assign  max real posit value if abs val is >= max real
      Fp = (Fp ^ -sign) + sign; 
    //   printf("the current output value is: %d \n", Fp);
      return Fp;
    }// printf("the current output value is: %d \n", Fp);
    if((v.si >= FLOAT_INF)){
      
      Fp ^= (Fp ^ _G_INFP) & -(v.si >= FLOAT_INF);                        // handles infinity
      Fp = (Fp ^ -sign) + sign;
      return Fp;
    }
    // printf("the current output value is: %d \n", v.si);
    // printf("the current output value is: %d \n", _G_MINREAL_INT);
    // printf("the current output value is: %d \n", _G_MINREALFP);
    if(f == 0){
        Fp ^= (Fp ^ _G_MINREALFP) & -(v.si != 0 && v.si <= _G_MINREAL_INT); // assigns  min real posit value if abs val is <= min real
        Fp ^= SIGN_MASK;
        return Fp;
    }
    if((v.si != 0 && v.si <= _G_MINREAL_INT)){
      
      Fp ^= (Fp ^ _G_MINREALFP) & -(v.si != 0 && v.si <= _G_MINREAL_INT); // assigns  min real posit value if abs val is <= min real
      
    //   Fp = (Fp ^ -sign) + sign;
      if(sign) Fp ^= SIGN_MASK;
      
      return Fp;
    }
    // printf("the current output value is: %d \n", Fp);
    // OBTAINING FRACTION
    TEMP_TYPE frac = v2.ui & FLOAT_FRACTION_MASK;
    frac = frac >> (23 - _G_FSIZE);
    
    // assemble
    fp32 temp_p = frac | regime_and_exp;
    // printf("the assembled value is: %d \n", temp_p);
    // round
    temp_p += (bool)(regime_and_exp & FPOSIT_HALFWAY_BIT_MASK) && ((temp_p & 1) | (regime_and_exp & FPOSIT_EXTRA_BITS_MASK));
    if (_G_NBITS != 16)
        temp_p <<= _G_FPOSIT_SHIFT_AMOUNT;
    // printf("the current temp_p value is: %d \n", temp_p);
    // printf("the current output value is: %d \n", temp_p);
    Fp ^= (temp_p ^ Fp) & -((v.f < _G_MAXREAL_INT) & (v.f < (-1 * _G_MINREAL_INT)));
    // printf("the current output value is: %d \n", Fp);
    // printf("the current output value is: %d \n", Fp);
    Fp = (Fp ^ -sign) + sign;
    // printf("result from the fp32 to fixed posit 16 function-----> the current output value is: %d \n", Fp);
    // corner cases

    // printf("the current output value is: %d \n", Fp);
    return Fp;
}

union ufloat
{
    float f;
    unsigned u;
};

int main()
{
    int nsize = 10;
    int es = 2;
    int rf = 2;
    float scale = 1.0;

    uint32_t int32_constants[16];
    uint64_t int64_constants[2];

    generate_fixedposit_constants(nsize, es, rf, int32_constants, int64_constants);
    float temp_input = 0.005*scale;
    ufloat tempf;
    tempf.f = temp_input;
    // printf("the float bit pattern is: %d \n", tempf.u);
    fp16 temp = fp32tofixedp16(temp_input, int32_constants, int64_constants);
    // fp16 temp = 32808;
    float output = fixedp16tofp32(temp, int32_constants, int64_constants);
    // printf("int32 constant\n");
    // for (int i = 0; i <16; i ++){
    //     printf("%d \n",int32_constants [i]);
    // }
    // printf("int64 constant\n");
    // for (int i = 0; i <2; i ++){
    //     printf("%lx \n",int64_constants [i]);
    // }
    union Bits max;
    union Bits min;
    max.ui = int32_constants[7];
    min.ui = int32_constants[8];
    printf("MaxReal %f MinReal %f \n", max.f, min.f);
    printf("input %f output %f \n", temp_input/scale, output/scale);
    // printf("temp %d \n", temp);
    /*
   for (int64_t i = 0; i < size; i++)
   {
     float temp_input = a_array[i]*scale;

     fp16 temp = fp32tofixedp16(temp_input, int32_constants, int64_constants);
     temp_input = fixedp16tofp32(temp, int32_constants, int64_constants);

     o_array[i] = temp_input/scale;

   }
     */
    return 0;
}

// int main() {
//     int nsize = 16;
//     int es = 2;
//     int rf = 3;
//     float scale = 1.0;

//     uint32_t int32_constants[16];
//     uint64_t int64_constants[2];

//     generate_fixedposit_constants(nsize, es, rf, int32_constants, int64_constants);

//     for (float temp_input = -3.0f; temp_input <= 3.0f; temp_input += 0.0625f) {
//         ufloat tempf;
//         tempf.f = temp_input;

//         uint16_t temp = fp32tofixedp16(temp_input, int32_constants, int64_constants);
//         float output = fixedp16tofp32(temp, int32_constants, int64_constants);

//         printf("input: %f output: %f\n", temp_input, output);
//     }

//     return 0;
// }