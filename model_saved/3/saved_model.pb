рс
КЏ
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
Т
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718бт
Д
conv2d_18/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_18/kernel
}
$conv2d_18/kernel/Read/ReadVariableOpReadVariableOpconv2d_18/kernel*&
_output_shapes
: *
dtype0
t
conv2d_18/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_18/bias
m
"conv2d_18/bias/Read/ReadVariableOpReadVariableOpconv2d_18/bias*
_output_shapes
: *
dtype0
Д
conv2d_19/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_19/kernel
}
$conv2d_19/kernel/Read/ReadVariableOpReadVariableOpconv2d_19/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_19/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_19/bias
m
"conv2d_19/bias/Read/ReadVariableOpReadVariableOpconv2d_19/bias*
_output_shapes
: *
dtype0
Д
conv2d_20/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_20/kernel
}
$conv2d_20/kernel/Read/ReadVariableOpReadVariableOpconv2d_20/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_20/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_20/bias
m
"conv2d_20/bias/Read/ReadVariableOpReadVariableOpconv2d_20/bias*
_output_shapes
: *
dtype0
Д
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0
Д
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
: *
dtype0
Д
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *!
shared_nameconv2d_23/kernel
}
$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*&
_output_shapes
:  *
dtype0
t
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_23/bias
m
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
_output_shapes
: *
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	А@*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:@*
dtype0
x
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*
shared_namedense_7/kernel
q
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel*
_output_shapes

:@*
dtype0
p
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
i
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h
VariableVarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable
a
Variable/Read/ReadVariableOpReadVariableOpVariable*
_output_shapes
:*
dtype0	
l

Variable_1VarHandleOp*
_output_shapes
: *
dtype0	*
shape:*
shared_name
Variable_1
e
Variable_1/Read/ReadVariableOpReadVariableOp
Variable_1*
_output_shapes
:*
dtype0	
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
Т
Adam/conv2d_18/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/m
Л
+Adam/conv2d_18/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_18/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/m
{
)Adam/conv2d_18/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_19/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_19/kernel/m
Л
+Adam/conv2d_19/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_19/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_19/bias/m
{
)Adam/conv2d_19/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_20/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_20/kernel/m
Л
+Adam/conv2d_20/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_20/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/m
{
)Adam/conv2d_20/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_21/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_21/kernel/m
Л
+Adam/conv2d_21/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_21/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/m
{
)Adam/conv2d_21/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_22/kernel/m
Л
+Adam/conv2d_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_22/bias/m
{
)Adam/conv2d_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/m*
_output_shapes
: *
dtype0
Т
Adam/conv2d_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_23/kernel/m
Л
+Adam/conv2d_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/m*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_23/bias/m
{
)Adam/conv2d_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/m*
_output_shapes
: *
dtype0
З
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_6/kernel/m
А
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	А@*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:@*
dtype0
Ж
Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m*
_output_shapes

:@*
dtype0
~
Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
w
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_18/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_18/kernel/v
Л
+Adam/conv2d_18/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_18/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_18/bias/v
{
)Adam/conv2d_18/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_18/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_19/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_19/kernel/v
Л
+Adam/conv2d_19/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_19/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_19/bias/v
{
)Adam/conv2d_19/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_19/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_20/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_20/kernel/v
Л
+Adam/conv2d_20/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_20/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_20/bias/v
{
)Adam/conv2d_20/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_20/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_21/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_21/kernel/v
Л
+Adam/conv2d_21/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_21/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_21/bias/v
{
)Adam/conv2d_21/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_21/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_22/kernel/v
Л
+Adam/conv2d_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_22/bias/v
{
)Adam/conv2d_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_22/bias/v*
_output_shapes
: *
dtype0
Т
Adam/conv2d_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *(
shared_nameAdam/conv2d_23/kernel/v
Л
+Adam/conv2d_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/kernel/v*&
_output_shapes
:  *
dtype0
В
Adam/conv2d_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_23/bias/v
{
)Adam/conv2d_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_23/bias/v*
_output_shapes
: *
dtype0
З
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А@*&
shared_nameAdam/dense_6/kernel/v
А
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	А@*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:@*
dtype0
Ж
Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v*
_output_shapes

:@*
dtype0
~
Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
w
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
≤j
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*нi
valueгiBаi Bўi
•
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
l
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
l
layer-0
layer-1
 regularization_losses
!	variables
"trainable_variables
#	keras_api
h

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
R
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
R
4regularization_losses
5	variables
6trainable_variables
7	keras_api
h

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
R
>regularization_losses
?	variables
@trainable_variables
A	keras_api
h

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
R
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
R
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
h

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
R
\regularization_losses
]	variables
^trainable_variables
_	keras_api
R
`regularization_losses
a	variables
btrainable_variables
c	keras_api
h

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
А
piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate$mв%mг.mд/mе8mж9mзBmиCmйLmкMmлVmмWmнdmоemпjmрkmс$vт%vу.vф/vх8vц9vчBvшCvщLvъMvыVvьWvэdvюev€jvАkvБ
 
v
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15
v
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15
≠
unon_trainable_variables
vlayer_metrics
regularization_losses
	variables
wmetrics
trainable_variables
xlayer_regularization_losses

ylayers
 

z	keras_api

{	keras_api
 
 
 
Ѓ
|non_trainable_variables
}layer_metrics
regularization_losses
	variables
~metrics
trainable_variables
layer_regularization_losses
Аlayers

	Б_rng
В	keras_api

	Г_rng
Д	keras_api
 
 
 
≤
Еnon_trainable_variables
Жlayer_metrics
 regularization_losses
!	variables
Зmetrics
"trainable_variables
 Иlayer_regularization_losses
Йlayers
\Z
VARIABLE_VALUEconv2d_18/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_18/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

$0
%1

$0
%1
≤
Кnon_trainable_variables
Лlayer_metrics
&regularization_losses
'	variables
Мmetrics
(trainable_variables
 Нlayer_regularization_losses
Оlayers
 
 
 
≤
Пnon_trainable_variables
Рlayer_metrics
*regularization_losses
+	variables
Сmetrics
,trainable_variables
 Тlayer_regularization_losses
Уlayers
\Z
VARIABLE_VALUEconv2d_19/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_19/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
≤
Фnon_trainable_variables
Хlayer_metrics
0regularization_losses
1	variables
Цmetrics
2trainable_variables
 Чlayer_regularization_losses
Шlayers
 
 
 
≤
Щnon_trainable_variables
Ъlayer_metrics
4regularization_losses
5	variables
Ыmetrics
6trainable_variables
 Ьlayer_regularization_losses
Эlayers
\Z
VARIABLE_VALUEconv2d_20/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_20/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

80
91

80
91
≤
Юnon_trainable_variables
Яlayer_metrics
:regularization_losses
;	variables
†metrics
<trainable_variables
 °layer_regularization_losses
Ґlayers
 
 
 
≤
£non_trainable_variables
§layer_metrics
>regularization_losses
?	variables
•metrics
@trainable_variables
 ¶layer_regularization_losses
Іlayers
\Z
VARIABLE_VALUEconv2d_21/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_21/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

B0
C1

B0
C1
≤
®non_trainable_variables
©layer_metrics
Dregularization_losses
E	variables
™metrics
Ftrainable_variables
 Ђlayer_regularization_losses
ђlayers
 
 
 
≤
≠non_trainable_variables
Ѓlayer_metrics
Hregularization_losses
I	variables
ѓmetrics
Jtrainable_variables
 ∞layer_regularization_losses
±layers
\Z
VARIABLE_VALUEconv2d_22/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_22/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
≤
≤non_trainable_variables
≥layer_metrics
Nregularization_losses
O	variables
іmetrics
Ptrainable_variables
 µlayer_regularization_losses
ґlayers
 
 
 
≤
Јnon_trainable_variables
Єlayer_metrics
Rregularization_losses
S	variables
єmetrics
Ttrainable_variables
 Їlayer_regularization_losses
їlayers
\Z
VARIABLE_VALUEconv2d_23/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_23/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

V0
W1

V0
W1
≤
Љnon_trainable_variables
љlayer_metrics
Xregularization_losses
Y	variables
Њmetrics
Ztrainable_variables
 њlayer_regularization_losses
јlayers
 
 
 
≤
Ѕnon_trainable_variables
¬layer_metrics
\regularization_losses
]	variables
√metrics
^trainable_variables
 ƒlayer_regularization_losses
≈layers
 
 
 
≤
∆non_trainable_variables
«layer_metrics
`regularization_losses
a	variables
»metrics
btrainable_variables
 …layer_regularization_losses
 layers
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
≤
Ћnon_trainable_variables
ћlayer_metrics
fregularization_losses
g	variables
Ќmetrics
htrainable_variables
 ќlayer_regularization_losses
ѕlayers
ZX
VARIABLE_VALUEdense_7/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_7/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
≤
–non_trainable_variables
—layer_metrics
lregularization_losses
m	variables
“metrics
ntrainable_variables
 ”layer_regularization_losses
‘layers
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

’0
÷1
 
~
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
 
 
 
 
 
 

0
1

„
_state_var
 

Ў
_state_var
 
 
 
 
 

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

ўtotal

Џcount
џ	variables
№	keras_api
I

Ёtotal

ёcount
я
_fn_kwargs
а	variables
б	keras_api
XV
VARIABLE_VALUEVariable:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUE
Variable_1:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

ў0
Џ1

џ	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

Ё0
ё1

а	variables
}
VARIABLE_VALUEAdam/conv2d_18/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_18/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_18/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_19/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_19/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_20/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_20/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_21/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_21/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_22/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_22/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_23/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_23/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_7/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_7/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Щ
"serving_default_sequential_5_inputPlaceholder*1
_output_shapes
:€€€€€€€€€АА*
dtype0*&
shape:€€€€€€€€€АА
г
StatefulPartitionedCallStatefulPartitionedCall"serving_default_sequential_5_inputconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *+
f&R$
"__inference_signature_wrapper_6666
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
г
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_18/kernel/Read/ReadVariableOp"conv2d_18/bias/Read/ReadVariableOp$conv2d_19/kernel/Read/ReadVariableOp"conv2d_19/bias/Read/ReadVariableOp$conv2d_20/kernel/Read/ReadVariableOp"conv2d_20/bias/Read/ReadVariableOp$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOpVariable/Read/ReadVariableOpVariable_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp+Adam/conv2d_18/kernel/m/Read/ReadVariableOp)Adam/conv2d_18/bias/m/Read/ReadVariableOp+Adam/conv2d_19/kernel/m/Read/ReadVariableOp)Adam/conv2d_19/bias/m/Read/ReadVariableOp+Adam/conv2d_20/kernel/m/Read/ReadVariableOp)Adam/conv2d_20/bias/m/Read/ReadVariableOp+Adam/conv2d_21/kernel/m/Read/ReadVariableOp)Adam/conv2d_21/bias/m/Read/ReadVariableOp+Adam/conv2d_22/kernel/m/Read/ReadVariableOp)Adam/conv2d_22/bias/m/Read/ReadVariableOp+Adam/conv2d_23/kernel/m/Read/ReadVariableOp)Adam/conv2d_23/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp+Adam/conv2d_18/kernel/v/Read/ReadVariableOp)Adam/conv2d_18/bias/v/Read/ReadVariableOp+Adam/conv2d_19/kernel/v/Read/ReadVariableOp)Adam/conv2d_19/bias/v/Read/ReadVariableOp+Adam/conv2d_20/kernel/v/Read/ReadVariableOp)Adam/conv2d_20/bias/v/Read/ReadVariableOp+Adam/conv2d_21/kernel/v/Read/ReadVariableOp)Adam/conv2d_21/bias/v/Read/ReadVariableOp+Adam/conv2d_22/kernel/v/Read/ReadVariableOp)Adam/conv2d_22/bias/v/Read/ReadVariableOp+Adam/conv2d_23/kernel/v/Read/ReadVariableOp)Adam/conv2d_23/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOpConst*H
TinA
?2=			*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *&
f!R
__inference__traced_save_8259
¬
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_18/kernelconv2d_18/biasconv2d_19/kernelconv2d_19/biasconv2d_20/kernelconv2d_20/biasconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateVariable
Variable_1totalcounttotal_1count_1Adam/conv2d_18/kernel/mAdam/conv2d_18/bias/mAdam/conv2d_19/kernel/mAdam/conv2d_19/bias/mAdam/conv2d_20/kernel/mAdam/conv2d_20/bias/mAdam/conv2d_21/kernel/mAdam/conv2d_21/bias/mAdam/conv2d_22/kernel/mAdam/conv2d_22/bias/mAdam/conv2d_23/kernel/mAdam/conv2d_23/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/conv2d_18/kernel/vAdam/conv2d_18/bias/vAdam/conv2d_19/kernel/vAdam/conv2d_19/bias/vAdam/conv2d_20/kernel/vAdam/conv2d_20/bias/vAdam/conv2d_21/kernel/vAdam/conv2d_21/bias/vAdam/conv2d_22/kernel/vAdam/conv2d_22/bias/vAdam/conv2d_23/kernel/vAdam/conv2d_23/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/v*G
Tin@
>2<*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *)
f$R"
 __inference__traced_restore_8446фн
Ф
ь
C__inference_conv2d_19_layer_call_and_return_conditional_losses_6100

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€}} 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
™
f
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_6032

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ї
»
+__inference_sequential_7_layer_call_fn_6740

inputs!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_62222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ѕ

b
F__inference_sequential_5_layer_call_and_return_conditional_losses_5568

inputs
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/size’
 resizing_1/resize/ResizeBilinearResizeBilinearinputsresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
≠

у
A__inference_dense_6_layer_call_and_return_conditional_losses_6198

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
ѓ«
о
F__inference_sequential_7_layer_call_and_return_conditional_losses_7132

inputsU
Gsequential_6_random_rotation_1_stateful_uniform_rngreadandskip_resource:	B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: B
(conv2d_19_conv2d_readvariableop_resource:  7
)conv2d_19_biasadd_readvariableop_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:  7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: 9
&dense_6_matmul_readvariableop_resource:	А@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ conv2d_18/BiasAdd/ReadVariableOpҐconv2d_18/Conv2D/ReadVariableOpҐ conv2d_19/BiasAdd/ReadVariableOpҐconv2d_19/Conv2D/ReadVariableOpҐ conv2d_20/BiasAdd/ReadVariableOpҐconv2d_20/Conv2D/ReadVariableOpҐ conv2d_21/BiasAdd/ReadVariableOpҐconv2d_21/Conv2D/ReadVariableOpҐ conv2d_22/BiasAdd/ReadVariableOpҐconv2d_22/Conv2D/ReadVariableOpҐ conv2d_23/BiasAdd/ReadVariableOpҐconv2d_23/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkipЫ
#sequential_5/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#sequential_5/resizing_1/resize/sizeь
-sequential_5/resizing_1/resize/ResizeBilinearResizeBilinearinputs,sequential_5/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2/
-sequential_5/resizing_1/resize/ResizeBilinearЗ
sequential_5/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2!
sequential_5/rescaling_1/Cast/xЛ
!sequential_5/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/rescaling_1/Cast_1/xй
sequential_5/rescaling_1/mulMul>sequential_5/resizing_1/resize/ResizeBilinear:resized_images:0(sequential_5/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/mulѕ
sequential_5/rescaling_1/addAddV2 sequential_5/rescaling_1/mul:z:0*sequential_5/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/addІ
Dsequential_6/random_flip_1/random_flip_left_right/control_dependencyIdentity sequential_5/rescaling_1/add:z:0*
T0*/
_class%
#!loc:@sequential_5/rescaling_1/add*1
_output_shapes
:€€€€€€€€€АА2F
Dsequential_6/random_flip_1/random_flip_left_right/control_dependencyп
7sequential_6/random_flip_1/random_flip_left_right/ShapeShapeMsequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:29
7sequential_6/random_flip_1/random_flip_left_right/ShapeЎ
Esequential_6/random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_6/random_flip_1/random_flip_left_right/strided_slice/stack№
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1№
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2О
?sequential_6/random_flip_1/random_flip_left_right/strided_sliceStridedSlice@sequential_6/random_flip_1/random_flip_left_right/Shape:output:0Nsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack:output:0Psequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Psequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_6/random_flip_1/random_flip_left_right/strided_sliceР
Fsequential_6/random_flip_1/random_flip_left_right/random_uniform/shapePackHsequential_6/random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2H
Fsequential_6/random_flip_1/random_flip_left_right/random_uniform/shape—
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/min—
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/maxљ
Nsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformOsequential_6/random_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02P
Nsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniformй
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/MulMulWsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0Msequential_6/random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/Mul»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3ж
?sequential_6/random_flip_1/random_flip_left_right/Reshape/shapePackHsequential_6/random_flip_1/random_flip_left_right/strided_slice:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?sequential_6/random_flip_1/random_flip_left_right/Reshape/shapeѕ
9sequential_6/random_flip_1/random_flip_left_right/ReshapeReshapeHsequential_6/random_flip_1/random_flip_left_right/random_uniform/Mul:z:0Hsequential_6/random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2;
9sequential_6/random_flip_1/random_flip_left_right/Reshapeщ
7sequential_6/random_flip_1/random_flip_left_right/RoundRoundBsequential_6/random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€29
7sequential_6/random_flip_1/random_flip_left_right/Roundќ
@sequential_6/random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_6/random_flip_1/random_flip_left_right/ReverseV2/axisЁ
;sequential_6/random_flip_1/random_flip_left_right/ReverseV2	ReverseV2Msequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0Isequential_6/random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2=
;sequential_6/random_flip_1/random_flip_left_right/ReverseV2і
5sequential_6/random_flip_1/random_flip_left_right/mulMul;sequential_6/random_flip_1/random_flip_left_right/Round:y:0Dsequential_6/random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА27
5sequential_6/random_flip_1/random_flip_left_right/mulЈ
7sequential_6/random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7sequential_6/random_flip_1/random_flip_left_right/sub/xЃ
5sequential_6/random_flip_1/random_flip_left_right/subSub@sequential_6/random_flip_1/random_flip_left_right/sub/x:output:0;sequential_6/random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€27
5sequential_6/random_flip_1/random_flip_left_right/subњ
7sequential_6/random_flip_1/random_flip_left_right/mul_1Mul9sequential_6/random_flip_1/random_flip_left_right/sub:z:0Msequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА29
7sequential_6/random_flip_1/random_flip_left_right/mul_1Ђ
5sequential_6/random_flip_1/random_flip_left_right/addAddV29sequential_6/random_flip_1/random_flip_left_right/mul:z:0;sequential_6/random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА27
5sequential_6/random_flip_1/random_flip_left_right/add”
Asequential_6/random_flip_1/random_flip_up_down/control_dependencyIdentity9sequential_6/random_flip_1/random_flip_left_right/add:z:0*
T0*H
_class>
<:loc:@sequential_6/random_flip_1/random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€АА2C
Asequential_6/random_flip_1/random_flip_up_down/control_dependencyж
4sequential_6/random_flip_1/random_flip_up_down/ShapeShapeJsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:26
4sequential_6/random_flip_1/random_flip_up_down/Shape“
Bsequential_6/random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack÷
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1÷
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2ь
<sequential_6/random_flip_1/random_flip_up_down/strided_sliceStridedSlice=sequential_6/random_flip_1/random_flip_up_down/Shape:output:0Ksequential_6/random_flip_1/random_flip_up_down/strided_slice/stack:output:0Msequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0Msequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/random_flip_1/random_flip_up_down/strided_sliceЗ
Csequential_6/random_flip_1/random_flip_up_down/random_uniform/shapePackEsequential_6/random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2E
Csequential_6/random_flip_1/random_flip_up_down/random_uniform/shapeЋ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/minЋ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/maxі
Ksequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniformLsequential_6/random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02M
Ksequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniformЁ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/MulMulTsequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0Jsequential_6/random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/Mul¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3‘
<sequential_6/random_flip_1/random_flip_up_down/Reshape/shapePackEsequential_6/random_flip_1/random_flip_up_down/strided_slice:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2>
<sequential_6/random_flip_1/random_flip_up_down/Reshape/shape√
6sequential_6/random_flip_1/random_flip_up_down/ReshapeReshapeEsequential_6/random_flip_1/random_flip_up_down/random_uniform/Mul:z:0Esequential_6/random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€28
6sequential_6/random_flip_1/random_flip_up_down/Reshapeр
4sequential_6/random_flip_1/random_flip_up_down/RoundRound?sequential_6/random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€26
4sequential_6/random_flip_1/random_flip_up_down/Round»
=sequential_6/random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential_6/random_flip_1/random_flip_up_down/ReverseV2/axis—
8sequential_6/random_flip_1/random_flip_up_down/ReverseV2	ReverseV2Jsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0Fsequential_6/random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2:
8sequential_6/random_flip_1/random_flip_up_down/ReverseV2®
2sequential_6/random_flip_1/random_flip_up_down/mulMul8sequential_6/random_flip_1/random_flip_up_down/Round:y:0Asequential_6/random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА24
2sequential_6/random_flip_1/random_flip_up_down/mul±
4sequential_6/random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4sequential_6/random_flip_1/random_flip_up_down/sub/xҐ
2sequential_6/random_flip_1/random_flip_up_down/subSub=sequential_6/random_flip_1/random_flip_up_down/sub/x:output:08sequential_6/random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€24
2sequential_6/random_flip_1/random_flip_up_down/sub≥
4sequential_6/random_flip_1/random_flip_up_down/mul_1Mul6sequential_6/random_flip_1/random_flip_up_down/sub:z:0Jsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА26
4sequential_6/random_flip_1/random_flip_up_down/mul_1Я
2sequential_6/random_flip_1/random_flip_up_down/addAddV26sequential_6/random_flip_1/random_flip_up_down/mul:z:08sequential_6/random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА24
2sequential_6/random_flip_1/random_flip_up_down/add≤
$sequential_6/random_rotation_1/ShapeShape6sequential_6/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2&
$sequential_6/random_rotation_1/Shape≤
2sequential_6/random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_6/random_rotation_1/strided_slice/stackґ
4sequential_6/random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice/stack_1ґ
4sequential_6/random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice/stack_2Ь
,sequential_6/random_rotation_1/strided_sliceStridedSlice-sequential_6/random_rotation_1/Shape:output:0;sequential_6/random_rotation_1/strided_slice/stack:output:0=sequential_6/random_rotation_1/strided_slice/stack_1:output:0=sequential_6/random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_6/random_rotation_1/strided_sliceґ
4sequential_6/random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice_1/stackЇ
6sequential_6/random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_1/stack_1Ї
6sequential_6/random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_1/stack_2¶
.sequential_6/random_rotation_1/strided_slice_1StridedSlice-sequential_6/random_rotation_1/Shape:output:0=sequential_6/random_rotation_1/strided_slice_1/stack:output:0?sequential_6/random_rotation_1/strided_slice_1/stack_1:output:0?sequential_6/random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_6/random_rotation_1/strided_slice_1ї
#sequential_6/random_rotation_1/CastCast7sequential_6/random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_6/random_rotation_1/Castґ
4sequential_6/random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice_2/stackЇ
6sequential_6/random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_2/stack_1Ї
6sequential_6/random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_2/stack_2¶
.sequential_6/random_rotation_1/strided_slice_2StridedSlice-sequential_6/random_rotation_1/Shape:output:0=sequential_6/random_rotation_1/strided_slice_2/stack:output:0?sequential_6/random_rotation_1/strided_slice_2/stack_1:output:0?sequential_6/random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_6/random_rotation_1/strided_slice_2њ
%sequential_6/random_rotation_1/Cast_1Cast7sequential_6/random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential_6/random_rotation_1/Cast_1џ
5sequential_6/random_rotation_1/stateful_uniform/shapePack5sequential_6/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:27
5sequential_6/random_rotation_1/stateful_uniform/shapeѓ
3sequential_6/random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њ25
3sequential_6/random_rotation_1/stateful_uniform/minѓ
3sequential_6/random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?25
3sequential_6/random_rotation_1/stateful_uniform/maxЄ
5sequential_6/random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_6/random_rotation_1/stateful_uniform/ConstХ
4sequential_6/random_rotation_1/stateful_uniform/ProdProd>sequential_6/random_rotation_1/stateful_uniform/shape:output:0>sequential_6/random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/stateful_uniform/Prod≤
6sequential_6/random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_6/random_rotation_1/stateful_uniform/Cast/xз
6sequential_6/random_rotation_1/stateful_uniform/Cast_1Cast=sequential_6/random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 28
6sequential_6/random_rotation_1/stateful_uniform/Cast_1ф
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkipGsequential_6_random_rotation_1_stateful_uniform_rngreadandskip_resource?sequential_6/random_rotation_1/stateful_uniform/Cast/x:output:0:sequential_6/random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:2@
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip‘
Csequential_6/random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_6/random_rotation_1/stateful_uniform/strided_slice/stackЎ
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1Ў
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2И
=sequential_6/random_rotation_1/stateful_uniform/strided_sliceStridedSliceFsequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Lsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack:output:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2?
=sequential_6/random_rotation_1/stateful_uniform/strided_sliceц
7sequential_6/random_rotation_1/stateful_uniform/BitcastBitcastFsequential_6/random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type029
7sequential_6/random_rotation_1/stateful_uniform/BitcastЎ
Esequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack№
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1№
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2А
?sequential_6/random_rotation_1/stateful_uniform/strided_slice_1StridedSliceFsequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Psequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Psequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2A
?sequential_6/random_rotation_1/stateful_uniform/strided_slice_1ь
9sequential_6/random_rotation_1/stateful_uniform/Bitcast_1BitcastHsequential_6/random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02;
9sequential_6/random_rotation_1/stateful_uniform/Bitcast_1ё
Lsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algт
Hsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2>sequential_6/random_rotation_1/stateful_uniform/shape:output:0Bsequential_6/random_rotation_1/stateful_uniform/Bitcast_1:output:0@sequential_6/random_rotation_1/stateful_uniform/Bitcast:output:0Usequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€2J
Hsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2О
3sequential_6/random_rotation_1/stateful_uniform/subSub<sequential_6/random_rotation_1/stateful_uniform/max:output:0<sequential_6/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 25
3sequential_6/random_rotation_1/stateful_uniform/subЂ
3sequential_6/random_rotation_1/stateful_uniform/mulMulQsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:07sequential_6/random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€25
3sequential_6/random_rotation_1/stateful_uniform/mulО
/sequential_6/random_rotation_1/stateful_uniformAdd7sequential_6/random_rotation_1/stateful_uniform/mul:z:0<sequential_6/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€21
/sequential_6/random_rotation_1/stateful_uniform±
4sequential_6/random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4sequential_6/random_rotation_1/rotation_matrix/sub/yъ
2sequential_6/random_rotation_1/rotation_matrix/subSub)sequential_6/random_rotation_1/Cast_1:y:0=sequential_6/random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 24
2sequential_6/random_rotation_1/rotation_matrix/sub“
2sequential_6/random_rotation_1/rotation_matrix/CosCos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/Cosµ
6sequential_6/random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_1/yА
4sequential_6/random_rotation_1/rotation_matrix/sub_1Sub)sequential_6/random_rotation_1/Cast_1:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_1П
2sequential_6/random_rotation_1/rotation_matrix/mulMul6sequential_6/random_rotation_1/rotation_matrix/Cos:y:08sequential_6/random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/mul“
2sequential_6/random_rotation_1/rotation_matrix/SinSin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/Sinµ
6sequential_6/random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_2/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_2Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_2У
4sequential_6/random_rotation_1/rotation_matrix/mul_1Mul6sequential_6/random_rotation_1/rotation_matrix/Sin:y:08sequential_6/random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_1У
4sequential_6/random_rotation_1/rotation_matrix/sub_3Sub6sequential_6/random_rotation_1/rotation_matrix/mul:z:08sequential_6/random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_3У
4sequential_6/random_rotation_1/rotation_matrix/sub_4Sub6sequential_6/random_rotation_1/rotation_matrix/sub:z:08sequential_6/random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_4є
8sequential_6/random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2:
8sequential_6/random_rotation_1/rotation_matrix/truediv/y¶
6sequential_6/random_rotation_1/rotation_matrix/truedivRealDiv8sequential_6/random_rotation_1/rotation_matrix/sub_4:z:0Asequential_6/random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€28
6sequential_6/random_rotation_1/rotation_matrix/truedivµ
6sequential_6/random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_5/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_5Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_5÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_1Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_1µ
6sequential_6/random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_6/yА
4sequential_6/random_rotation_1/rotation_matrix/sub_6Sub)sequential_6/random_rotation_1/Cast_1:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_6Х
4sequential_6/random_rotation_1/rotation_matrix/mul_2Mul8sequential_6/random_rotation_1/rotation_matrix/Sin_1:y:08sequential_6/random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_2÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_1Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_1µ
6sequential_6/random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_7/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_7Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_7Х
4sequential_6/random_rotation_1/rotation_matrix/mul_3Mul8sequential_6/random_rotation_1/rotation_matrix/Cos_1:y:08sequential_6/random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_3У
2sequential_6/random_rotation_1/rotation_matrix/addAddV28sequential_6/random_rotation_1/rotation_matrix/mul_2:z:08sequential_6/random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/addУ
4sequential_6/random_rotation_1/rotation_matrix/sub_8Sub8sequential_6/random_rotation_1/rotation_matrix/sub_5:z:06sequential_6/random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_8љ
:sequential_6/random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:sequential_6/random_rotation_1/rotation_matrix/truediv_1/yђ
8sequential_6/random_rotation_1/rotation_matrix/truediv_1RealDiv8sequential_6/random_rotation_1/rotation_matrix/sub_8:z:0Csequential_6/random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2:
8sequential_6/random_rotation_1/rotation_matrix/truediv_1ѕ
4sequential_6/random_rotation_1/rotation_matrix/ShapeShape3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:26
4sequential_6/random_rotation_1/rotation_matrix/Shape“
Bsequential_6/random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack÷
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1÷
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2ь
<sequential_6/random_rotation_1/rotation_matrix/strided_sliceStridedSlice=sequential_6/random_rotation_1/rotation_matrix/Shape:output:0Ksequential_6/random_rotation_1/rotation_matrix/strided_slice/stack:output:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/random_rotation_1/rotation_matrix/strided_slice÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_2Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_2Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_1StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Cos_2:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_1÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_2Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_2Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_2StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Sin_2:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_2к
2sequential_6/random_rotation_1/rotation_matrix/NegNegGsequential_6/random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/NegЁ
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2≥
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_3StridedSlice:sequential_6/random_rotation_1/rotation_matrix/truediv:z:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_3÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_3Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_3Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_4StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Sin_3:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_4÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_3Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_3Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_5StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Cos_3:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_5Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2µ
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_6StridedSlice<sequential_6/random_rotation_1/rotation_matrix/truediv_1:z:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_6Ї
:sequential_6/random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_6/random_rotation_1/rotation_matrix/zeros/mul/y®
8sequential_6/random_rotation_1/rotation_matrix/zeros/mulMulEsequential_6/random_rotation_1/rotation_matrix/strided_slice:output:0Csequential_6/random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_6/random_rotation_1/rotation_matrix/zeros/mulљ
;sequential_6/random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2=
;sequential_6/random_rotation_1/rotation_matrix/zeros/Less/y£
9sequential_6/random_rotation_1/rotation_matrix/zeros/LessLess<sequential_6/random_rotation_1/rotation_matrix/zeros/mul:z:0Dsequential_6/random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_6/random_rotation_1/rotation_matrix/zeros/Lessј
=sequential_6/random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_6/random_rotation_1/rotation_matrix/zeros/packed/1њ
;sequential_6/random_rotation_1/rotation_matrix/zeros/packedPackEsequential_6/random_rotation_1/rotation_matrix/strided_slice:output:0Fsequential_6/random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_6/random_rotation_1/rotation_matrix/zeros/packedљ
:sequential_6/random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_6/random_rotation_1/rotation_matrix/zeros/Const±
4sequential_6/random_rotation_1/rotation_matrix/zerosFillDsequential_6/random_rotation_1/rotation_matrix/zeros/packed:output:0Csequential_6/random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/zerosЇ
:sequential_6/random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_6/random_rotation_1/rotation_matrix/concat/axisё
5sequential_6/random_rotation_1/rotation_matrix/concatConcatV2Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_1:output:06sequential_6/random_rotation_1/rotation_matrix/Neg:y:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_3:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_4:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_5:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_6:output:0=sequential_6/random_rotation_1/rotation_matrix/zeros:output:0Csequential_6/random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€27
5sequential_6/random_rotation_1/rotation_matrix/concat∆
.sequential_6/random_rotation_1/transform/ShapeShape6sequential_6/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:20
.sequential_6/random_rotation_1/transform/Shape∆
<sequential_6/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_6/random_rotation_1/transform/strided_slice/stack 
>sequential_6/random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/random_rotation_1/transform/strided_slice/stack_1 
>sequential_6/random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/random_rotation_1/transform/strided_slice/stack_2ƒ
6sequential_6/random_rotation_1/transform/strided_sliceStridedSlice7sequential_6/random_rotation_1/transform/Shape:output:0Esequential_6/random_rotation_1/transform/strided_slice/stack:output:0Gsequential_6/random_rotation_1/transform/strided_slice/stack_1:output:0Gsequential_6/random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:28
6sequential_6/random_rotation_1/transform/strided_sliceѓ
3sequential_6/random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_6/random_rotation_1/transform/fill_valueФ
Csequential_6/random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36sequential_6/random_flip_1/random_flip_up_down/add:z:0>sequential_6/random_rotation_1/rotation_matrix/concat:output:0?sequential_6/random_rotation_1/transform/strided_slice:output:0<sequential_6/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2E
Csequential_6/random_rotation_1/transform/ImageProjectiveTransformV3≥
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpЦ
conv2d_18/Conv2DConv2DXsequential_6/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
conv2d_18/Conv2D™
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp≤
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/BiasAddА
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/Relu 
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool≥
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpЁ
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
conv2d_19/Conv2D™
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp∞
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/Relu 
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*/
_output_shapes
:€€€€€€€€€>> *
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool≥
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_20/Conv2D/ReadVariableOpЁ
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
conv2d_20/Conv2D™
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp∞
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/Relu 
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool≥
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_21/Conv2D/ReadVariableOpЁ
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_21/Conv2D™
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp∞
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/Relu 
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool≥
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpЁ
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_22/Conv2D™
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp∞
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/Relu 
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool≥
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpЁ
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_23/Conv2D™
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp∞
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/Relu 
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
flatten_3/Const°
flatten_3/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_3/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/Softmaxќ
IdentityIdentitydense_7/Softmax:softmax:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp?^sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€АА: : : : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2А
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ѓ
Ћ
"__inference_signature_wrapper_6666
sequential_5_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallЦ
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *(
f#R!
__inference__wrapped_model_55542
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
’
K
/__inference_max_pooling2d_20_layer_call_fn_6026

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_60202
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ё
‘
+__inference_sequential_7_layer_call_fn_6703
sequential_5_input!
unknown: 
	unknown_0: #
	unknown_1:  
	unknown_2: #
	unknown_3:  
	unknown_4: #
	unknown_5:  
	unknown_6: #
	unknown_7:  
	unknown_8: #
	unknown_9:  

unknown_10: 

unknown_11:	А@

unknown_12:@

unknown_13:@

unknown_14:
identityИҐStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_62222
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
ЯЃ
г
F__inference_sequential_6_layer_call_and_return_conditional_losses_5804

inputsH
:random_rotation_1_stateful_uniform_rngreadandskip_resource:	
identityИҐ1random_rotation_1/stateful_uniform/RngReadAndSkipЁ
7random_flip_1/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:€€€€€€€€€АА29
7random_flip_1/random_flip_left_right/control_dependency»
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeЊ
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stack¬
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1¬
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2ј
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceй
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЈ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЈ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7random_flip_1/random_flip_left_right/random_uniform/maxЦ
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformµ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€29
7random_flip_1/random_flip_left_right/random_uniform/MulЃ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3Ш
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shapeЫ
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2.
,random_flip_1/random_flip_left_right/Reshape“
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*random_flip_1/random_flip_left_right/Roundі
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axis©
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА20
.random_flip_1/random_flip_left_right/ReverseV2А
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/mulЭ
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*random_flip_1/random_flip_left_right/sub/xъ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(random_flip_1/random_flip_left_right/subЛ
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2,
*random_flip_1/random_flip_left_right/mul_1ч
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/addЯ
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€АА26
4random_flip_1/random_flip_up_down/control_dependencyњ
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeЄ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackЉ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1Љ
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ѓ
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceа
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shape±
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/min±
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4random_flip_1/random_flip_up_down/random_uniform/maxН
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniform©
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€26
4random_flip_1/random_flip_up_down/random_uniform/Mul®
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1®
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2®
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3Ж
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shapeП
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2+
)random_flip_1/random_flip_up_down/Reshape…
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'random_flip_1/random_flip_up_down/RoundЃ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axisЭ
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2-
+random_flip_1/random_flip_up_down/ReverseV2ф
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/mulЧ
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_flip_1/random_flip_up_down/sub/xо
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%random_flip_1/random_flip_up_down/sub€
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2)
'random_flip_1/random_flip_up_down/mul_1л
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/addЛ
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/ShapeШ
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stackЬ
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1Ь
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2ќ
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_sliceЬ
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack†
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1†
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2Ў
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1Ф
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/CastЬ
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack†
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1†
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2Ў
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2Ш
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1і
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shapeХ
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њ2(
&random_rotation_1/stateful_uniform/minХ
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?2(
&random_rotation_1/stateful_uniform/maxЮ
(random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(random_rotation_1/stateful_uniform/Constб
'random_rotation_1/stateful_uniform/ProdProd1random_rotation_1/stateful_uniform/shape:output:01random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/stateful_uniform/ProdШ
)random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_rotation_1/stateful_uniform/Cast/xј
)random_rotation_1/stateful_uniform/Cast_1Cast0random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)random_rotation_1/stateful_uniform/Cast_1≥
1random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_1_stateful_uniform_rngreadandskip_resource2random_rotation_1/stateful_uniform/Cast/x:output:0-random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:23
1random_rotation_1/stateful_uniform/RngReadAndSkipЇ
6random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_1/stateful_uniform/strided_slice/stackЊ
8random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_1Њ
8random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_2Ї
0random_rotation_1/stateful_uniform/strided_sliceStridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0?random_rotation_1/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask22
0random_rotation_1/stateful_uniform/strided_sliceѕ
*random_rotation_1/stateful_uniform/BitcastBitcast9random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation_1/stateful_uniform/BitcastЊ
8random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice_1/stack¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2≤
2random_rotation_1/stateful_uniform/strided_slice_1StridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:24
2random_rotation_1/stateful_uniform/strided_slice_1’
,random_rotation_1/stateful_uniform/Bitcast_1Bitcast;random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02.
,random_rotation_1/stateful_uniform/Bitcast_1ƒ
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2A
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg§
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_1/stateful_uniform/shape:output:05random_rotation_1/stateful_uniform/Bitcast_1:output:03random_rotation_1/stateful_uniform/Bitcast:output:0Hrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€2=
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2Џ
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subч
&random_rotation_1/stateful_uniform/mulMulDrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€2(
&random_rotation_1/stateful_uniform/mulЏ
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€2$
"random_rotation_1/stateful_uniformЧ
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_rotation_1/rotation_matrix/sub/y∆
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЂ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/CosЫ
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_1/yћ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1џ
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/mulЂ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/SinЫ
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_2/y 
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2я
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_1я
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_3я
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_4Я
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yт
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2+
)random_rotation_1/rotation_matrix/truedivЫ
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_5/y 
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5ѓ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_1Ы
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_6/yћ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6б
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_2ѓ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_1Ы
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_7/y 
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7б
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_3я
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/addя
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_8£
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yш
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2-
+random_rotation_1/rotation_matrix/truediv_1®
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeЄ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackЉ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1Љ
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ѓ
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceѓ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_2√
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stack«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1ѓ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_2√
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stack«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2√
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/Neg√
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stack«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2е
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3ѓ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_3√
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stack«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4ѓ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_3√
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stack«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5√
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stack«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2з
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6†
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yф
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mul£
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и20
.random_rotation_1/rotation_matrix/zeros/Less/yп
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/Less¶
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1Л
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packed£
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Constэ
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/zeros†
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axis№
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2*
(random_rotation_1/rotation_matrix/concatЯ
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/Shapeђ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stack∞
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1∞
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2ц
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_sliceХ
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_value∆
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ё
IdentityIdentityKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:02^random_rotation_1/stateful_uniform/RngReadAndSkip*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2f
1random_rotation_1/stateful_uniform/RngReadAndSkip1random_rotation_1/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
€f
в
F__inference_sequential_7_layer_call_and_return_conditional_losses_7206
sequential_5_inputB
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: B
(conv2d_19_conv2d_readvariableop_resource:  7
)conv2d_19_biasadd_readvariableop_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:  7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: 9
&dense_6_matmul_readvariableop_resource:	А@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ conv2d_18/BiasAdd/ReadVariableOpҐconv2d_18/Conv2D/ReadVariableOpҐ conv2d_19/BiasAdd/ReadVariableOpҐconv2d_19/Conv2D/ReadVariableOpҐ conv2d_20/BiasAdd/ReadVariableOpҐconv2d_20/Conv2D/ReadVariableOpҐ conv2d_21/BiasAdd/ReadVariableOpҐconv2d_21/Conv2D/ReadVariableOpҐ conv2d_22/BiasAdd/ReadVariableOpҐconv2d_22/Conv2D/ReadVariableOpҐ conv2d_23/BiasAdd/ReadVariableOpҐconv2d_23/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpЫ
#sequential_5/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#sequential_5/resizing_1/resize/sizeИ
-sequential_5/resizing_1/resize/ResizeBilinearResizeBilinearsequential_5_input,sequential_5/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2/
-sequential_5/resizing_1/resize/ResizeBilinearЗ
sequential_5/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2!
sequential_5/rescaling_1/Cast/xЛ
!sequential_5/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/rescaling_1/Cast_1/xй
sequential_5/rescaling_1/mulMul>sequential_5/resizing_1/resize/ResizeBilinear:resized_images:0(sequential_5/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/mulѕ
sequential_5/rescaling_1/addAddV2 sequential_5/rescaling_1/mul:z:0*sequential_5/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/add≥
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpё
conv2d_18/Conv2DConv2D sequential_5/rescaling_1/add:z:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
conv2d_18/Conv2D™
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp≤
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/BiasAddА
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/Relu 
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool≥
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpЁ
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
conv2d_19/Conv2D™
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp∞
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/Relu 
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*/
_output_shapes
:€€€€€€€€€>> *
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool≥
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_20/Conv2D/ReadVariableOpЁ
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
conv2d_20/Conv2D™
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp∞
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/Relu 
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool≥
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_21/Conv2D/ReadVariableOpЁ
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_21/Conv2D™
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp∞
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/Relu 
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool≥
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpЁ
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_22/Conv2D™
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp∞
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/Relu 
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool≥
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpЁ
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_23/Conv2D™
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp∞
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/Relu 
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
flatten_3/Const°
flatten_3/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_3/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/SoftmaxН
IdentityIdentitydense_7/Softmax:softmax:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
’
K
/__inference_max_pooling2d_19_layer_call_fn_6014

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_60082
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
я

l
F__inference_sequential_5_layer_call_and_return_conditional_losses_7506
resizing_1_input
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/sizeя
 resizing_1/resize/ResizeBilinearResizeBilinearresizing_1_inputresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
≠v
В
__inference__traced_save_8259
file_prefix/
+savev2_conv2d_18_kernel_read_readvariableop-
)savev2_conv2d_18_bias_read_readvariableop/
+savev2_conv2d_19_kernel_read_readvariableop-
)savev2_conv2d_19_bias_read_readvariableop/
+savev2_conv2d_20_kernel_read_readvariableop-
)savev2_conv2d_20_bias_read_readvariableop/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop'
#savev2_variable_read_readvariableop	)
%savev2_variable_1_read_readvariableop	$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop6
2savev2_adam_conv2d_18_kernel_m_read_readvariableop4
0savev2_adam_conv2d_18_bias_m_read_readvariableop6
2savev2_adam_conv2d_19_kernel_m_read_readvariableop4
0savev2_adam_conv2d_19_bias_m_read_readvariableop6
2savev2_adam_conv2d_20_kernel_m_read_readvariableop4
0savev2_adam_conv2d_20_bias_m_read_readvariableop6
2savev2_adam_conv2d_21_kernel_m_read_readvariableop4
0savev2_adam_conv2d_21_bias_m_read_readvariableop6
2savev2_adam_conv2d_22_kernel_m_read_readvariableop4
0savev2_adam_conv2d_22_bias_m_read_readvariableop6
2savev2_adam_conv2d_23_kernel_m_read_readvariableop4
0savev2_adam_conv2d_23_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop6
2savev2_adam_conv2d_18_kernel_v_read_readvariableop4
0savev2_adam_conv2d_18_bias_v_read_readvariableop6
2savev2_adam_conv2d_19_kernel_v_read_readvariableop4
0savev2_adam_conv2d_19_bias_v_read_readvariableop6
2savev2_adam_conv2d_20_kernel_v_read_readvariableop4
0savev2_adam_conv2d_20_bias_v_read_readvariableop6
2savev2_adam_conv2d_21_kernel_v_read_readvariableop4
0savev2_adam_conv2d_21_bias_v_read_readvariableop6
2savev2_adam_conv2d_22_kernel_v_read_readvariableop4
0savev2_adam_conv2d_22_bias_v_read_readvariableop6
2savev2_adam_conv2d_23_kernel_v_read_readvariableop4
0savev2_adam_conv2d_23_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename¶!
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Є 
valueЃ BЂ <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesГ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Н
valueГBА<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices°
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_18_kernel_read_readvariableop)savev2_conv2d_18_bias_read_readvariableop+savev2_conv2d_19_kernel_read_readvariableop)savev2_conv2d_19_bias_read_readvariableop+savev2_conv2d_20_kernel_read_readvariableop)savev2_conv2d_20_bias_read_readvariableop+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop#savev2_variable_read_readvariableop%savev2_variable_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop2savev2_adam_conv2d_18_kernel_m_read_readvariableop0savev2_adam_conv2d_18_bias_m_read_readvariableop2savev2_adam_conv2d_19_kernel_m_read_readvariableop0savev2_adam_conv2d_19_bias_m_read_readvariableop2savev2_adam_conv2d_20_kernel_m_read_readvariableop0savev2_adam_conv2d_20_bias_m_read_readvariableop2savev2_adam_conv2d_21_kernel_m_read_readvariableop0savev2_adam_conv2d_21_bias_m_read_readvariableop2savev2_adam_conv2d_22_kernel_m_read_readvariableop0savev2_adam_conv2d_22_bias_m_read_readvariableop2savev2_adam_conv2d_23_kernel_m_read_readvariableop0savev2_adam_conv2d_23_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop2savev2_adam_conv2d_18_kernel_v_read_readvariableop0savev2_adam_conv2d_18_bias_v_read_readvariableop2savev2_adam_conv2d_19_kernel_v_read_readvariableop0savev2_adam_conv2d_19_bias_v_read_readvariableop2savev2_adam_conv2d_20_kernel_v_read_readvariableop0savev2_adam_conv2d_20_bias_v_read_readvariableop2savev2_adam_conv2d_21_kernel_v_read_readvariableop0savev2_adam_conv2d_21_bias_v_read_readvariableop2savev2_adam_conv2d_22_kernel_v_read_readvariableop0savev2_adam_conv2d_22_bias_v_read_readvariableop2savev2_adam_conv2d_23_kernel_v_read_readvariableop0savev2_adam_conv2d_23_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *J
dtypes@
>2<			2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0* 
_input_shapesЄ
µ: : : :  : :  : :  : :  : :  : :	А@:@:@:: : : : : ::: : : : : : :  : :  : :  : :  : :  : :	А@:@:@:: : :  : :  : :  : :  : :  : :	А@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :,	(
&
_output_shapes
:  : 


_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :%!

_output_shapes
:	А@: 

_output_shapes
:@:$ 

_output_shapes

:@: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:  : 

_output_shapes
: :, (
&
_output_shapes
:  : !

_output_shapes
: :,"(
&
_output_shapes
:  : #

_output_shapes
: :,$(
&
_output_shapes
:  : %

_output_shapes
: :,&(
&
_output_shapes
:  : '

_output_shapes
: :%(!

_output_shapes
:	А@: )

_output_shapes
:@:$* 

_output_shapes

:@: +

_output_shapes
::,,(
&
_output_shapes
: : -

_output_shapes
: :,.(
&
_output_shapes
:  : /

_output_shapes
: :,0(
&
_output_shapes
:  : 1

_output_shapes
: :,2(
&
_output_shapes
:  : 3

_output_shapes
: :,4(
&
_output_shapes
:  : 5

_output_shapes
: :,6(
&
_output_shapes
:  : 7

_output_shapes
: :%8!

_output_shapes
:	А@: 9

_output_shapes
:@:$: 

_output_shapes

:@: ;

_output_shapes
::<

_output_shapes
: 
ђь
ў$
 __inference__traced_restore_8446
file_prefix;
!assignvariableop_conv2d_18_kernel: /
!assignvariableop_1_conv2d_18_bias: =
#assignvariableop_2_conv2d_19_kernel:  /
!assignvariableop_3_conv2d_19_bias: =
#assignvariableop_4_conv2d_20_kernel:  /
!assignvariableop_5_conv2d_20_bias: =
#assignvariableop_6_conv2d_21_kernel:  /
!assignvariableop_7_conv2d_21_bias: =
#assignvariableop_8_conv2d_22_kernel:  /
!assignvariableop_9_conv2d_22_bias: >
$assignvariableop_10_conv2d_23_kernel:  0
"assignvariableop_11_conv2d_23_bias: 5
"assignvariableop_12_dense_6_kernel:	А@.
 assignvariableop_13_dense_6_bias:@4
"assignvariableop_14_dense_7_kernel:@.
 assignvariableop_15_dense_7_bias:'
assignvariableop_16_adam_iter:	 )
assignvariableop_17_adam_beta_1: )
assignvariableop_18_adam_beta_2: (
assignvariableop_19_adam_decay: 0
&assignvariableop_20_adam_learning_rate: *
assignvariableop_21_variable:	,
assignvariableop_22_variable_1:	#
assignvariableop_23_total: #
assignvariableop_24_count: %
assignvariableop_25_total_1: %
assignvariableop_26_count_1: E
+assignvariableop_27_adam_conv2d_18_kernel_m: 7
)assignvariableop_28_adam_conv2d_18_bias_m: E
+assignvariableop_29_adam_conv2d_19_kernel_m:  7
)assignvariableop_30_adam_conv2d_19_bias_m: E
+assignvariableop_31_adam_conv2d_20_kernel_m:  7
)assignvariableop_32_adam_conv2d_20_bias_m: E
+assignvariableop_33_adam_conv2d_21_kernel_m:  7
)assignvariableop_34_adam_conv2d_21_bias_m: E
+assignvariableop_35_adam_conv2d_22_kernel_m:  7
)assignvariableop_36_adam_conv2d_22_bias_m: E
+assignvariableop_37_adam_conv2d_23_kernel_m:  7
)assignvariableop_38_adam_conv2d_23_bias_m: <
)assignvariableop_39_adam_dense_6_kernel_m:	А@5
'assignvariableop_40_adam_dense_6_bias_m:@;
)assignvariableop_41_adam_dense_7_kernel_m:@5
'assignvariableop_42_adam_dense_7_bias_m:E
+assignvariableop_43_adam_conv2d_18_kernel_v: 7
)assignvariableop_44_adam_conv2d_18_bias_v: E
+assignvariableop_45_adam_conv2d_19_kernel_v:  7
)assignvariableop_46_adam_conv2d_19_bias_v: E
+assignvariableop_47_adam_conv2d_20_kernel_v:  7
)assignvariableop_48_adam_conv2d_20_bias_v: E
+assignvariableop_49_adam_conv2d_21_kernel_v:  7
)assignvariableop_50_adam_conv2d_21_bias_v: E
+assignvariableop_51_adam_conv2d_22_kernel_v:  7
)assignvariableop_52_adam_conv2d_22_bias_v: E
+assignvariableop_53_adam_conv2d_23_kernel_v:  7
)assignvariableop_54_adam_conv2d_23_bias_v: <
)assignvariableop_55_adam_dense_6_kernel_v:	А@5
'assignvariableop_56_adam_dense_6_bias_v:@;
)assignvariableop_57_adam_dense_7_kernel_v:@5
'assignvariableop_58_adam_dense_7_bias_v:
identity_60ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_33ҐAssignVariableOp_34ҐAssignVariableOp_35ҐAssignVariableOp_36ҐAssignVariableOp_37ҐAssignVariableOp_38ҐAssignVariableOp_39ҐAssignVariableOp_4ҐAssignVariableOp_40ҐAssignVariableOp_41ҐAssignVariableOp_42ҐAssignVariableOp_43ҐAssignVariableOp_44ҐAssignVariableOp_45ҐAssignVariableOp_46ҐAssignVariableOp_47ҐAssignVariableOp_48ҐAssignVariableOp_49ҐAssignVariableOp_5ҐAssignVariableOp_50ҐAssignVariableOp_51ҐAssignVariableOp_52ҐAssignVariableOp_53ҐAssignVariableOp_54ҐAssignVariableOp_55ҐAssignVariableOp_56ҐAssignVariableOp_57ҐAssignVariableOp_58ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9ђ!
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Є 
valueЃ BЂ <B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-0/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB:layer-1/layer-1/_rng/_state_var/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesЙ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:<*
dtype0*Н
valueГBА<B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЏ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ж
_output_shapesу
р::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*J
dtypes@
>2<			2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_18_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_18_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_19_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¶
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_19_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4®
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_20_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5¶
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_20_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6®
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_21_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¶
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_21_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8®
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_22_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¶
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_22_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ђ
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_23_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11™
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_23_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12™
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_6_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13®
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_6_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14™
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_7_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15®
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_7_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_16•
AssignVariableOp_16AssignVariableOpassignvariableop_16_adam_iterIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17І
AssignVariableOp_17AssignVariableOpassignvariableop_17_adam_beta_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18І
AssignVariableOp_18AssignVariableOpassignvariableop_18_adam_beta_2Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¶
AssignVariableOp_19AssignVariableOpassignvariableop_19_adam_decayIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20Ѓ
AssignVariableOp_20AssignVariableOp&assignvariableop_20_adam_learning_rateIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21§
AssignVariableOp_21AssignVariableOpassignvariableop_21_variableIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_22¶
AssignVariableOp_22AssignVariableOpassignvariableop_22_variable_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23°
AssignVariableOp_23AssignVariableOpassignvariableop_23_totalIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24°
AssignVariableOp_24AssignVariableOpassignvariableop_24_countIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25£
AssignVariableOp_25AssignVariableOpassignvariableop_25_total_1Identity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26£
AssignVariableOp_26AssignVariableOpassignvariableop_26_count_1Identity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27≥
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_18_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_18_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≥
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_19_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_19_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31≥
AssignVariableOp_31AssignVariableOp+assignvariableop_31_adam_conv2d_20_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32±
AssignVariableOp_32AssignVariableOp)assignvariableop_32_adam_conv2d_20_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33≥
AssignVariableOp_33AssignVariableOp+assignvariableop_33_adam_conv2d_21_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34±
AssignVariableOp_34AssignVariableOp)assignvariableop_34_adam_conv2d_21_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35≥
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_22_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36±
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_22_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37≥
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_23_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38±
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_23_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40ѓ
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_7_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42ѓ
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_7_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43≥
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_18_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44±
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_18_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45≥
AssignVariableOp_45AssignVariableOp+assignvariableop_45_adam_conv2d_19_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46±
AssignVariableOp_46AssignVariableOp)assignvariableop_46_adam_conv2d_19_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47≥
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_20_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_20_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49≥
AssignVariableOp_49AssignVariableOp+assignvariableop_49_adam_conv2d_21_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_conv2d_21_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51≥
AssignVariableOp_51AssignVariableOp+assignvariableop_51_adam_conv2d_22_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_conv2d_22_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53≥
AssignVariableOp_53AssignVariableOp+assignvariableop_53_adam_conv2d_23_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_conv2d_23_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55±
AssignVariableOp_55AssignVariableOp)assignvariableop_55_adam_dense_6_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56ѓ
AssignVariableOp_56AssignVariableOp'assignvariableop_56_adam_dense_6_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57±
AssignVariableOp_57AssignVariableOp)assignvariableop_57_adam_dense_7_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58ѓ
AssignVariableOp_58AssignVariableOp'assignvariableop_58_adam_dense_7_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_589
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpр

Identity_59Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_59г

Identity_60IdentityIdentity_59:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_60"#
identity_60Identity_60:output:0*Л
_input_shapesz
x: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
™
f
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_6020

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
 J
т
F__inference_sequential_7_layer_call_and_return_conditional_losses_6437

inputs
sequential_6_6386:	(
conv2d_18_6389: 
conv2d_18_6391: (
conv2d_19_6395:  
conv2d_19_6397: (
conv2d_20_6401:  
conv2d_20_6403: (
conv2d_21_6407:  
conv2d_21_6409: (
conv2d_22_6413:  
conv2d_22_6415: (
conv2d_23_6419:  
conv2d_23_6421: 
dense_6_6426:	А@
dense_6_6428:@
dense_7_6431:@
dense_7_6433:
identityИҐ!conv2d_18/StatefulPartitionedCallҐ!conv2d_19/StatefulPartitionedCallҐ!conv2d_20/StatefulPartitionedCallҐ!conv2d_21/StatefulPartitionedCallҐ!conv2d_22/StatefulPartitionedCallҐ!conv2d_23/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallҐ$sequential_6/StatefulPartitionedCallи
sequential_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55882
sequential_5/PartitionedCall≥
$sequential_6/StatefulPartitionedCallStatefulPartitionedCall%sequential_5/PartitionedCall:output:0sequential_6_6386*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_58042&
$sequential_6/StatefulPartitionedCallƒ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall-sequential_6/StatefulPartitionedCall:output:0conv2d_18_6389conv2d_18_6391*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€юю *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_60822#
!conv2d_18/StatefulPartitionedCallЦ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_59962"
 max_pooling2d_18/PartitionedCallЊ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_19_6395conv2d_19_6397*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_61002#
!conv2d_19/StatefulPartitionedCallЦ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_60082"
 max_pooling2d_19/PartitionedCallЊ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_20_6401conv2d_20_6403*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€<< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_61182#
!conv2d_20/StatefulPartitionedCallЦ
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_60202"
 max_pooling2d_20/PartitionedCallЊ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_21_6407conv2d_21_6409*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_61362#
!conv2d_21/StatefulPartitionedCallЦ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_60322"
 max_pooling2d_21/PartitionedCallЊ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_6413conv2d_22_6415*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_61542#
!conv2d_22/StatefulPartitionedCallЦ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_60442"
 max_pooling2d_22/PartitionedCallЊ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_6419conv2d_23_6421*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_61722#
!conv2d_23/StatefulPartitionedCallЦ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_60562"
 max_pooling2d_23/PartitionedCallщ
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_61852
flatten_3/PartitionedCall•
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_6426dense_6_6428*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_61982!
dense_6/StatefulPartitionedCallЂ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_6431dense_7_6433*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_62152!
dense_7/StatefulPartitionedCallњ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall%^sequential_6/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€АА: : : : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2L
$sequential_6/StatefulPartitionedCall$sequential_6/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
я

l
F__inference_sequential_5_layer_call_and_return_conditional_losses_7496
resizing_1_input
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/sizeя
 resizing_1/resize/ResizeBilinearResizeBilinearresizing_1_inputresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
Ф
ь
C__inference_conv2d_23_layer_call_and_return_conditional_losses_7998

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
†
ь
C__inference_conv2d_18_layer_call_and_return_conditional_losses_6082

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
Relu°
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€юю 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
b
F__inference_sequential_6_layer_call_and_return_conditional_losses_7534

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ѕ

b
F__inference_sequential_5_layer_call_and_return_conditional_losses_7476

inputs
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/size’
 resizing_1/resize/ResizeBilinearResizeBilinearinputsresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
П
T
+__inference_sequential_6_layer_call_fn_7511
random_flip_1_input
identityџ
PartitionedCallPartitionedCallrandom_flip_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_56242
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
ъА
џ
__inference__wrapped_model_5554
sequential_5_inputO
5sequential_7_conv2d_18_conv2d_readvariableop_resource: D
6sequential_7_conv2d_18_biasadd_readvariableop_resource: O
5sequential_7_conv2d_19_conv2d_readvariableop_resource:  D
6sequential_7_conv2d_19_biasadd_readvariableop_resource: O
5sequential_7_conv2d_20_conv2d_readvariableop_resource:  D
6sequential_7_conv2d_20_biasadd_readvariableop_resource: O
5sequential_7_conv2d_21_conv2d_readvariableop_resource:  D
6sequential_7_conv2d_21_biasadd_readvariableop_resource: O
5sequential_7_conv2d_22_conv2d_readvariableop_resource:  D
6sequential_7_conv2d_22_biasadd_readvariableop_resource: O
5sequential_7_conv2d_23_conv2d_readvariableop_resource:  D
6sequential_7_conv2d_23_biasadd_readvariableop_resource: F
3sequential_7_dense_6_matmul_readvariableop_resource:	А@B
4sequential_7_dense_6_biasadd_readvariableop_resource:@E
3sequential_7_dense_7_matmul_readvariableop_resource:@B
4sequential_7_dense_7_biasadd_readvariableop_resource:
identityИҐ-sequential_7/conv2d_18/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_18/Conv2D/ReadVariableOpҐ-sequential_7/conv2d_19/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_19/Conv2D/ReadVariableOpҐ-sequential_7/conv2d_20/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_20/Conv2D/ReadVariableOpҐ-sequential_7/conv2d_21/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_21/Conv2D/ReadVariableOpҐ-sequential_7/conv2d_22/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_22/Conv2D/ReadVariableOpҐ-sequential_7/conv2d_23/BiasAdd/ReadVariableOpҐ,sequential_7/conv2d_23/Conv2D/ReadVariableOpҐ+sequential_7/dense_6/BiasAdd/ReadVariableOpҐ*sequential_7/dense_6/MatMul/ReadVariableOpҐ+sequential_7/dense_7/BiasAdd/ReadVariableOpҐ*sequential_7/dense_7/MatMul/ReadVariableOpµ
0sequential_7/sequential_5/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      22
0sequential_7/sequential_5/resizing_1/resize/sizeѓ
:sequential_7/sequential_5/resizing_1/resize/ResizeBilinearResizeBilinearsequential_5_input9sequential_7/sequential_5/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2<
:sequential_7/sequential_5/resizing_1/resize/ResizeBilinear°
,sequential_7/sequential_5/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2.
,sequential_7/sequential_5/rescaling_1/Cast/x•
.sequential_7/sequential_5/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.sequential_7/sequential_5/rescaling_1/Cast_1/xЭ
)sequential_7/sequential_5/rescaling_1/mulMulKsequential_7/sequential_5/resizing_1/resize/ResizeBilinear:resized_images:05sequential_7/sequential_5/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2+
)sequential_7/sequential_5/rescaling_1/mulГ
)sequential_7/sequential_5/rescaling_1/addAddV2-sequential_7/sequential_5/rescaling_1/mul:z:07sequential_7/sequential_5/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2+
)sequential_7/sequential_5/rescaling_1/addЏ
,sequential_7/conv2d_18/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02.
,sequential_7/conv2d_18/Conv2D/ReadVariableOpТ
sequential_7/conv2d_18/Conv2DConv2D-sequential_7/sequential_5/rescaling_1/add:z:04sequential_7/conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
sequential_7/conv2d_18/Conv2D—
-sequential_7/conv2d_18/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_18/BiasAdd/ReadVariableOpж
sequential_7/conv2d_18/BiasAddBiasAdd&sequential_7/conv2d_18/Conv2D:output:05sequential_7/conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2 
sequential_7/conv2d_18/BiasAddІ
sequential_7/conv2d_18/ReluRelu'sequential_7/conv2d_18/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
sequential_7/conv2d_18/Reluс
%sequential_7/max_pooling2d_18/MaxPoolMaxPool)sequential_7/conv2d_18/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_18/MaxPoolЏ
,sequential_7/conv2d_19/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_7/conv2d_19/Conv2D/ReadVariableOpС
sequential_7/conv2d_19/Conv2DConv2D.sequential_7/max_pooling2d_18/MaxPool:output:04sequential_7/conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
sequential_7/conv2d_19/Conv2D—
-sequential_7/conv2d_19/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_19/BiasAdd/ReadVariableOpд
sequential_7/conv2d_19/BiasAddBiasAdd&sequential_7/conv2d_19/Conv2D:output:05sequential_7/conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2 
sequential_7/conv2d_19/BiasAdd•
sequential_7/conv2d_19/ReluRelu'sequential_7/conv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
sequential_7/conv2d_19/Reluс
%sequential_7/max_pooling2d_19/MaxPoolMaxPool)sequential_7/conv2d_19/Relu:activations:0*/
_output_shapes
:€€€€€€€€€>> *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_19/MaxPoolЏ
,sequential_7/conv2d_20/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_7/conv2d_20/Conv2D/ReadVariableOpС
sequential_7/conv2d_20/Conv2DConv2D.sequential_7/max_pooling2d_19/MaxPool:output:04sequential_7/conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
sequential_7/conv2d_20/Conv2D—
-sequential_7/conv2d_20/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_20/BiasAdd/ReadVariableOpд
sequential_7/conv2d_20/BiasAddBiasAdd&sequential_7/conv2d_20/Conv2D:output:05sequential_7/conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2 
sequential_7/conv2d_20/BiasAdd•
sequential_7/conv2d_20/ReluRelu'sequential_7/conv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
sequential_7/conv2d_20/Reluс
%sequential_7/max_pooling2d_20/MaxPoolMaxPool)sequential_7/conv2d_20/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_20/MaxPoolЏ
,sequential_7/conv2d_21/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_7/conv2d_21/Conv2D/ReadVariableOpС
sequential_7/conv2d_21/Conv2DConv2D.sequential_7/max_pooling2d_20/MaxPool:output:04sequential_7/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential_7/conv2d_21/Conv2D—
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_21/BiasAdd/ReadVariableOpд
sequential_7/conv2d_21/BiasAddBiasAdd&sequential_7/conv2d_21/Conv2D:output:05sequential_7/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2 
sequential_7/conv2d_21/BiasAdd•
sequential_7/conv2d_21/ReluRelu'sequential_7/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_7/conv2d_21/Reluс
%sequential_7/max_pooling2d_21/MaxPoolMaxPool)sequential_7/conv2d_21/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_21/MaxPoolЏ
,sequential_7/conv2d_22/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_7/conv2d_22/Conv2D/ReadVariableOpС
sequential_7/conv2d_22/Conv2DConv2D.sequential_7/max_pooling2d_21/MaxPool:output:04sequential_7/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential_7/conv2d_22/Conv2D—
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_22/BiasAdd/ReadVariableOpд
sequential_7/conv2d_22/BiasAddBiasAdd&sequential_7/conv2d_22/Conv2D:output:05sequential_7/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2 
sequential_7/conv2d_22/BiasAdd•
sequential_7/conv2d_22/ReluRelu'sequential_7/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_7/conv2d_22/Reluс
%sequential_7/max_pooling2d_22/MaxPoolMaxPool)sequential_7/conv2d_22/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_22/MaxPoolЏ
,sequential_7/conv2d_23/Conv2D/ReadVariableOpReadVariableOp5sequential_7_conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02.
,sequential_7/conv2d_23/Conv2D/ReadVariableOpС
sequential_7/conv2d_23/Conv2DConv2D.sequential_7/max_pooling2d_22/MaxPool:output:04sequential_7/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
sequential_7/conv2d_23/Conv2D—
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_7_conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02/
-sequential_7/conv2d_23/BiasAdd/ReadVariableOpд
sequential_7/conv2d_23/BiasAddBiasAdd&sequential_7/conv2d_23/Conv2D:output:05sequential_7/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2 
sequential_7/conv2d_23/BiasAdd•
sequential_7/conv2d_23/ReluRelu'sequential_7/conv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
sequential_7/conv2d_23/Reluс
%sequential_7/max_pooling2d_23/MaxPoolMaxPool)sequential_7/conv2d_23/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2'
%sequential_7/max_pooling2d_23/MaxPoolН
sequential_7/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
sequential_7/flatten_3/Const’
sequential_7/flatten_3/ReshapeReshape.sequential_7/max_pooling2d_23/MaxPool:output:0%sequential_7/flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2 
sequential_7/flatten_3/ReshapeЌ
*sequential_7/dense_6/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_6_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02,
*sequential_7/dense_6/MatMul/ReadVariableOp”
sequential_7/dense_6/MatMulMatMul'sequential_7/flatten_3/Reshape:output:02sequential_7/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_7/dense_6/MatMulЋ
+sequential_7/dense_6/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+sequential_7/dense_6/BiasAdd/ReadVariableOp’
sequential_7/dense_6/BiasAddBiasAdd%sequential_7/dense_6/MatMul:product:03sequential_7/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_7/dense_6/BiasAddЧ
sequential_7/dense_6/ReluRelu%sequential_7/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
sequential_7/dense_6/Reluћ
*sequential_7/dense_7/MatMul/ReadVariableOpReadVariableOp3sequential_7_dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02,
*sequential_7/dense_7/MatMul/ReadVariableOp”
sequential_7/dense_7/MatMulMatMul'sequential_7/dense_6/Relu:activations:02sequential_7/dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_7/dense_7/MatMulЋ
+sequential_7/dense_7/BiasAdd/ReadVariableOpReadVariableOp4sequential_7_dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+sequential_7/dense_7/BiasAdd/ReadVariableOp’
sequential_7/dense_7/BiasAddBiasAdd%sequential_7/dense_7/MatMul:product:03sequential_7/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_7/dense_7/BiasAdd†
sequential_7/dense_7/SoftmaxSoftmax%sequential_7/dense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
sequential_7/dense_7/Softmaxк
IdentityIdentity&sequential_7/dense_7/Softmax:softmax:0.^sequential_7/conv2d_18/BiasAdd/ReadVariableOp-^sequential_7/conv2d_18/Conv2D/ReadVariableOp.^sequential_7/conv2d_19/BiasAdd/ReadVariableOp-^sequential_7/conv2d_19/Conv2D/ReadVariableOp.^sequential_7/conv2d_20/BiasAdd/ReadVariableOp-^sequential_7/conv2d_20/Conv2D/ReadVariableOp.^sequential_7/conv2d_21/BiasAdd/ReadVariableOp-^sequential_7/conv2d_21/Conv2D/ReadVariableOp.^sequential_7/conv2d_22/BiasAdd/ReadVariableOp-^sequential_7/conv2d_22/Conv2D/ReadVariableOp.^sequential_7/conv2d_23/BiasAdd/ReadVariableOp-^sequential_7/conv2d_23/Conv2D/ReadVariableOp,^sequential_7/dense_6/BiasAdd/ReadVariableOp+^sequential_7/dense_6/MatMul/ReadVariableOp,^sequential_7/dense_7/BiasAdd/ReadVariableOp+^sequential_7/dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 2^
-sequential_7/conv2d_18/BiasAdd/ReadVariableOp-sequential_7/conv2d_18/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_18/Conv2D/ReadVariableOp,sequential_7/conv2d_18/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_19/BiasAdd/ReadVariableOp-sequential_7/conv2d_19/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_19/Conv2D/ReadVariableOp,sequential_7/conv2d_19/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_20/BiasAdd/ReadVariableOp-sequential_7/conv2d_20/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_20/Conv2D/ReadVariableOp,sequential_7/conv2d_20/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_21/BiasAdd/ReadVariableOp-sequential_7/conv2d_21/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_21/Conv2D/ReadVariableOp,sequential_7/conv2d_21/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_22/BiasAdd/ReadVariableOp-sequential_7/conv2d_22/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_22/Conv2D/ReadVariableOp,sequential_7/conv2d_22/Conv2D/ReadVariableOp2^
-sequential_7/conv2d_23/BiasAdd/ReadVariableOp-sequential_7/conv2d_23/BiasAdd/ReadVariableOp2\
,sequential_7/conv2d_23/Conv2D/ReadVariableOp,sequential_7/conv2d_23/Conv2D/ReadVariableOp2Z
+sequential_7/dense_6/BiasAdd/ReadVariableOp+sequential_7/dense_6/BiasAdd/ReadVariableOp2X
*sequential_7/dense_6/MatMul/ReadVariableOp*sequential_7/dense_6/MatMul/ReadVariableOp2Z
+sequential_7/dense_7/BiasAdd/ReadVariableOp+sequential_7/dense_7/BiasAdd/ReadVariableOp2X
*sequential_7/dense_7/MatMul/ReadVariableOp*sequential_7/dense_7/MatMul/ReadVariableOp:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
’
K
/__inference_max_pooling2d_21_layer_call_fn_6038

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_60322
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7918

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€}} 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
«
o
F__inference_sequential_6_layer_call_and_return_conditional_losses_7708
random_flip_1_input
identityq
IdentityIdentityrandom_flip_1_input*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
’
K
/__inference_max_pooling2d_18_layer_call_fn_6002

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_59962
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
г
в
+__inference_sequential_7_layer_call_fn_6779

inputs
unknown:	#
	unknown_0: 
	unknown_1: #
	unknown_2:  
	unknown_3: #
	unknown_4:  
	unknown_5: #
	unknown_6:  
	unknown_7: #
	unknown_8:  
	unknown_9: $

unknown_10:  

unknown_11: 

unknown_12:	А@

unknown_13:@

unknown_14:@

unknown_15:
identityИҐStatefulPartitionedCallЊ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_64372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€АА: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
ћ
D
(__inference_flatten_3_layer_call_fn_8003

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_61852
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_23_layer_call_and_return_conditional_losses_6172

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
„H
™
F__inference_sequential_7_layer_call_and_return_conditional_losses_6222

inputs(
conv2d_18_6083: 
conv2d_18_6085: (
conv2d_19_6101:  
conv2d_19_6103: (
conv2d_20_6119:  
conv2d_20_6121: (
conv2d_21_6137:  
conv2d_21_6139: (
conv2d_22_6155:  
conv2d_22_6157: (
conv2d_23_6173:  
conv2d_23_6175: 
dense_6_6199:	А@
dense_6_6201:@
dense_7_6216:@
dense_7_6218:
identityИҐ!conv2d_18/StatefulPartitionedCallҐ!conv2d_19/StatefulPartitionedCallҐ!conv2d_20/StatefulPartitionedCallҐ!conv2d_21/StatefulPartitionedCallҐ!conv2d_22/StatefulPartitionedCallҐ!conv2d_23/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐdense_7/StatefulPartitionedCallи
sequential_5/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55682
sequential_5/PartitionedCallЗ
sequential_6/PartitionedCallPartitionedCall%sequential_5/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_56242
sequential_6/PartitionedCallЉ
!conv2d_18/StatefulPartitionedCallStatefulPartitionedCall%sequential_6/PartitionedCall:output:0conv2d_18_6083conv2d_18_6085*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€юю *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_60822#
!conv2d_18/StatefulPartitionedCallЦ
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_18/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_59962"
 max_pooling2d_18/PartitionedCallЊ
!conv2d_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0conv2d_19_6101conv2d_19_6103*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_61002#
!conv2d_19/StatefulPartitionedCallЦ
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_19/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€>> * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_60082"
 max_pooling2d_19/PartitionedCallЊ
!conv2d_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0conv2d_20_6119conv2d_20_6121*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€<< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_61182#
!conv2d_20/StatefulPartitionedCallЦ
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_60202"
 max_pooling2d_20/PartitionedCallЊ
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0conv2d_21_6137conv2d_21_6139*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_61362#
!conv2d_21/StatefulPartitionedCallЦ
 max_pooling2d_21/PartitionedCallPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_60322"
 max_pooling2d_21/PartitionedCallЊ
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_21/PartitionedCall:output:0conv2d_22_6155conv2d_22_6157*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_61542#
!conv2d_22/StatefulPartitionedCallЦ
 max_pooling2d_22/PartitionedCallPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_60442"
 max_pooling2d_22/PartitionedCallЊ
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_22/PartitionedCall:output:0conv2d_23_6173conv2d_23_6175*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_61722#
!conv2d_23/StatefulPartitionedCallЦ
 max_pooling2d_23/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_60562"
 max_pooling2d_23/PartitionedCallщ
flatten_3/PartitionedCallPartitionedCall)max_pooling2d_23/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_flatten_3_layer_call_and_return_conditional_losses_61852
flatten_3/PartitionedCall•
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_3/PartitionedCall:output:0dense_6_6199dense_6_6201*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_61982!
dense_6/StatefulPartitionedCallЂ
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_6216dense_7_6218*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_62152!
dense_7/StatefulPartitionedCallШ
IdentityIdentity(dense_7/StatefulPartitionedCall:output:0"^conv2d_18/StatefulPartitionedCall"^conv2d_19/StatefulPartitionedCall"^conv2d_20/StatefulPartitionedCall"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 2F
!conv2d_18/StatefulPartitionedCall!conv2d_18/StatefulPartitionedCall2F
!conv2d_19/StatefulPartitionedCall!conv2d_19/StatefulPartitionedCall2F
!conv2d_20/StatefulPartitionedCall!conv2d_20/StatefulPartitionedCall2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
™
f
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_6044

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ч
{
+__inference_sequential_6_layer_call_fn_7523

inputs
unknown:	
identityИҐStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_58042
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_22_layer_call_and_return_conditional_losses_6154

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
З
о
+__inference_sequential_7_layer_call_fn_6818
sequential_5_input
unknown:	#
	unknown_0: 
	unknown_1: #
	unknown_2:  
	unknown_3: #
	unknown_4:  
	unknown_5: #
	unknown_6:  
	unknown_7: #
	unknown_8:  
	unknown_9: $

unknown_10:  

unknown_11: 

unknown_12:	А@

unknown_13:@

unknown_14:@

unknown_15:
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallsequential_5_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*2
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_7_layer_call_and_return_conditional_losses_64372
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€АА: : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
ЯЃ
г
F__inference_sequential_6_layer_call_and_return_conditional_losses_7704

inputsH
:random_rotation_1_stateful_uniform_rngreadandskip_resource:	
identityИҐ1random_rotation_1/stateful_uniform/RngReadAndSkipЁ
7random_flip_1/random_flip_left_right/control_dependencyIdentityinputs*
T0*
_class
loc:@inputs*1
_output_shapes
:€€€€€€€€€АА29
7random_flip_1/random_flip_left_right/control_dependency»
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeЊ
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stack¬
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1¬
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2ј
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceй
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЈ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЈ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7random_flip_1/random_flip_left_right/random_uniform/maxЦ
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformµ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€29
7random_flip_1/random_flip_left_right/random_uniform/MulЃ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3Ш
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shapeЫ
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2.
,random_flip_1/random_flip_left_right/Reshape“
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*random_flip_1/random_flip_left_right/Roundі
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axis©
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА20
.random_flip_1/random_flip_left_right/ReverseV2А
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/mulЭ
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*random_flip_1/random_flip_left_right/sub/xъ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(random_flip_1/random_flip_left_right/subЛ
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2,
*random_flip_1/random_flip_left_right/mul_1ч
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/addЯ
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€АА26
4random_flip_1/random_flip_up_down/control_dependencyњ
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeЄ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackЉ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1Љ
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ѓ
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceа
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shape±
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/min±
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4random_flip_1/random_flip_up_down/random_uniform/maxН
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniform©
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€26
4random_flip_1/random_flip_up_down/random_uniform/Mul®
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1®
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2®
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3Ж
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shapeП
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2+
)random_flip_1/random_flip_up_down/Reshape…
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'random_flip_1/random_flip_up_down/RoundЃ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axisЭ
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2-
+random_flip_1/random_flip_up_down/ReverseV2ф
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/mulЧ
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_flip_1/random_flip_up_down/sub/xо
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%random_flip_1/random_flip_up_down/sub€
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2)
'random_flip_1/random_flip_up_down/mul_1л
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/addЛ
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/ShapeШ
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stackЬ
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1Ь
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2ќ
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_sliceЬ
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack†
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1†
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2Ў
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1Ф
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/CastЬ
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack†
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1†
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2Ў
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2Ш
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1і
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shapeХ
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њ2(
&random_rotation_1/stateful_uniform/minХ
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?2(
&random_rotation_1/stateful_uniform/maxЮ
(random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(random_rotation_1/stateful_uniform/Constб
'random_rotation_1/stateful_uniform/ProdProd1random_rotation_1/stateful_uniform/shape:output:01random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/stateful_uniform/ProdШ
)random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_rotation_1/stateful_uniform/Cast/xј
)random_rotation_1/stateful_uniform/Cast_1Cast0random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)random_rotation_1/stateful_uniform/Cast_1≥
1random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_1_stateful_uniform_rngreadandskip_resource2random_rotation_1/stateful_uniform/Cast/x:output:0-random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:23
1random_rotation_1/stateful_uniform/RngReadAndSkipЇ
6random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_1/stateful_uniform/strided_slice/stackЊ
8random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_1Њ
8random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_2Ї
0random_rotation_1/stateful_uniform/strided_sliceStridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0?random_rotation_1/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask22
0random_rotation_1/stateful_uniform/strided_sliceѕ
*random_rotation_1/stateful_uniform/BitcastBitcast9random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation_1/stateful_uniform/BitcastЊ
8random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice_1/stack¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2≤
2random_rotation_1/stateful_uniform/strided_slice_1StridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:24
2random_rotation_1/stateful_uniform/strided_slice_1’
,random_rotation_1/stateful_uniform/Bitcast_1Bitcast;random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02.
,random_rotation_1/stateful_uniform/Bitcast_1ƒ
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2A
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg§
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_1/stateful_uniform/shape:output:05random_rotation_1/stateful_uniform/Bitcast_1:output:03random_rotation_1/stateful_uniform/Bitcast:output:0Hrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€2=
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2Џ
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subч
&random_rotation_1/stateful_uniform/mulMulDrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€2(
&random_rotation_1/stateful_uniform/mulЏ
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€2$
"random_rotation_1/stateful_uniformЧ
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_rotation_1/rotation_matrix/sub/y∆
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЂ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/CosЫ
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_1/yћ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1џ
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/mulЂ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/SinЫ
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_2/y 
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2я
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_1я
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_3я
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_4Я
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yт
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2+
)random_rotation_1/rotation_matrix/truedivЫ
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_5/y 
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5ѓ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_1Ы
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_6/yћ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6б
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_2ѓ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_1Ы
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_7/y 
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7б
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_3я
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/addя
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_8£
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yш
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2-
+random_rotation_1/rotation_matrix/truediv_1®
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeЄ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackЉ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1Љ
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ѓ
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceѓ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_2√
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stack«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1ѓ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_2√
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stack«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2√
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/Neg√
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stack«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2е
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3ѓ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_3√
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stack«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4ѓ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_3√
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stack«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5√
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stack«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2з
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6†
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yф
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mul£
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и20
.random_rotation_1/rotation_matrix/zeros/Less/yп
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/Less¶
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1Л
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packed£
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Constэ
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/zeros†
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axis№
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2*
(random_rotation_1/rotation_matrix/concatЯ
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/Shapeђ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stack∞
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1∞
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2ц
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_sliceХ
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_value∆
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ё
IdentityIdentityKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:02^random_rotation_1/stateful_uniform/RngReadAndSkip*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2f
1random_rotation_1/stateful_uniform/RngReadAndSkip1random_rotation_1/stateful_uniform/RngReadAndSkip:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
±

т
A__inference_dense_7_layer_call_and_return_conditional_losses_6215

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
Ѕ
Э
(__inference_conv2d_21_layer_call_fn_7947

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_21_layer_call_and_return_conditional_losses_61362
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ
Э
(__inference_conv2d_23_layer_call_fn_7987

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_23_layer_call_and_return_conditional_losses_61722
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_20_layer_call_and_return_conditional_losses_7938

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€<< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€>> 
 
_user_specified_nameinputs
и
G
+__inference_sequential_5_layer_call_fn_7456

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55682
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Х
У
&__inference_dense_7_layer_call_fn_8038

inputs
unknown:@
	unknown_0:
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_7_layer_call_and_return_conditional_losses_62152
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
и
G
+__inference_sequential_6_layer_call_fn_7516

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_56242
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
b
F__inference_sequential_6_layer_call_and_return_conditional_losses_5624

inputs
identityd
IdentityIdentityinputs*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
≠

у
A__inference_dense_6_layer_call_and_return_conditional_losses_8029

inputs1
matmul_readvariableop_resource:	А@-
biasadd_readvariableop_resource:@
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
ReluЧ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
Ж
Q
+__inference_sequential_5_layer_call_fn_7466
resizing_1_input
identityЎ
PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55882
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
и
G
+__inference_sequential_5_layer_call_fn_7461

inputs
identityќ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55882
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
™
f
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_5996

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
е
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_6185

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
џf
÷
F__inference_sequential_7_layer_call_and_return_conditional_losses_6892

inputsB
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: B
(conv2d_19_conv2d_readvariableop_resource:  7
)conv2d_19_biasadd_readvariableop_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:  7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: 9
&dense_6_matmul_readvariableop_resource:	А@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ conv2d_18/BiasAdd/ReadVariableOpҐconv2d_18/Conv2D/ReadVariableOpҐ conv2d_19/BiasAdd/ReadVariableOpҐconv2d_19/Conv2D/ReadVariableOpҐ conv2d_20/BiasAdd/ReadVariableOpҐconv2d_20/Conv2D/ReadVariableOpҐ conv2d_21/BiasAdd/ReadVariableOpҐconv2d_21/Conv2D/ReadVariableOpҐ conv2d_22/BiasAdd/ReadVariableOpҐconv2d_22/Conv2D/ReadVariableOpҐ conv2d_23/BiasAdd/ReadVariableOpҐconv2d_23/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpЫ
#sequential_5/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#sequential_5/resizing_1/resize/sizeь
-sequential_5/resizing_1/resize/ResizeBilinearResizeBilinearinputs,sequential_5/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2/
-sequential_5/resizing_1/resize/ResizeBilinearЗ
sequential_5/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2!
sequential_5/rescaling_1/Cast/xЛ
!sequential_5/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/rescaling_1/Cast_1/xй
sequential_5/rescaling_1/mulMul>sequential_5/resizing_1/resize/ResizeBilinear:resized_images:0(sequential_5/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/mulѕ
sequential_5/rescaling_1/addAddV2 sequential_5/rescaling_1/mul:z:0*sequential_5/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/add≥
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpё
conv2d_18/Conv2DConv2D sequential_5/rescaling_1/add:z:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
conv2d_18/Conv2D™
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp≤
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/BiasAddА
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/Relu 
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool≥
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpЁ
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
conv2d_19/Conv2D™
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp∞
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/Relu 
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*/
_output_shapes
:€€€€€€€€€>> *
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool≥
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_20/Conv2D/ReadVariableOpЁ
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
conv2d_20/Conv2D™
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp∞
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/Relu 
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool≥
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_21/Conv2D/ReadVariableOpЁ
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_21/Conv2D™
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp∞
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/Relu 
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool≥
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpЁ
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_22/Conv2D™
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp∞
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/Relu 
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool≥
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpЁ
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_23/Conv2D™
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp∞
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/Relu 
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
flatten_3/Const°
flatten_3/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_3/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/SoftmaxН
IdentityIdentitydense_7/Softmax:softmax:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:€€€€€€€€€АА: : : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
±

т
A__inference_dense_7_layer_call_and_return_conditional_losses_8049

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2	
SoftmaxЦ
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:€€€€€€€€€@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:€€€€€€€€€@
 
_user_specified_nameinputs
™
f
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_6008

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѕ

b
F__inference_sequential_5_layer_call_and_return_conditional_losses_5588

inputs
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/size’
 resizing_1/resize/ResizeBilinearResizeBilinearinputsresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
њ
И
+__inference_sequential_6_layer_call_fn_7530
random_flip_1_input
unknown:	
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallrandom_flip_1_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_6_layer_call_and_return_conditional_losses_58042
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 22
StatefulPartitionedCallStatefulPartitionedCall:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
Ѕ
Э
(__inference_conv2d_19_layer_call_fn_7907

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€}} *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_19_layer_call_and_return_conditional_losses_61002
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€}} 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
”Ѓ
р
F__inference_sequential_6_layer_call_and_return_conditional_losses_7878
random_flip_1_inputH
:random_rotation_1_stateful_uniform_rngreadandskip_resource:	
identityИҐ1random_rotation_1/stateful_uniform/RngReadAndSkipч
7random_flip_1/random_flip_left_right/control_dependencyIdentityrandom_flip_1_input*
T0*&
_class
loc:@random_flip_1_input*1
_output_shapes
:€€€€€€€€€АА29
7random_flip_1/random_flip_left_right/control_dependency»
*random_flip_1/random_flip_left_right/ShapeShape@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:2,
*random_flip_1/random_flip_left_right/ShapeЊ
8random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8random_flip_1/random_flip_left_right/strided_slice/stack¬
:random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_1¬
:random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_flip_1/random_flip_left_right/strided_slice/stack_2ј
2random_flip_1/random_flip_left_right/strided_sliceStridedSlice3random_flip_1/random_flip_left_right/Shape:output:0Arandom_flip_1/random_flip_left_right/strided_slice/stack:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Crandom_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2random_flip_1/random_flip_left_right/strided_sliceй
9random_flip_1/random_flip_left_right/random_uniform/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2;
9random_flip_1/random_flip_left_right/random_uniform/shapeЈ
7random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    29
7random_flip_1/random_flip_left_right/random_uniform/minЈ
7random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7random_flip_1/random_flip_left_right/random_uniform/maxЦ
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformBrandom_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02C
Arandom_flip_1/random_flip_left_right/random_uniform/RandomUniformµ
7random_flip_1/random_flip_left_right/random_uniform/MulMulJrandom_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0@random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€29
7random_flip_1/random_flip_left_right/random_uniform/MulЃ
4random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/1Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/2Ѓ
4random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :26
4random_flip_1/random_flip_left_right/Reshape/shape/3Ш
2random_flip_1/random_flip_left_right/Reshape/shapePack;random_flip_1/random_flip_left_right/strided_slice:output:0=random_flip_1/random_flip_left_right/Reshape/shape/1:output:0=random_flip_1/random_flip_left_right/Reshape/shape/2:output:0=random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:24
2random_flip_1/random_flip_left_right/Reshape/shapeЫ
,random_flip_1/random_flip_left_right/ReshapeReshape;random_flip_1/random_flip_left_right/random_uniform/Mul:z:0;random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2.
,random_flip_1/random_flip_left_right/Reshape“
*random_flip_1/random_flip_left_right/RoundRound5random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2,
*random_flip_1/random_flip_left_right/Roundі
3random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:25
3random_flip_1/random_flip_left_right/ReverseV2/axis©
.random_flip_1/random_flip_left_right/ReverseV2	ReverseV2@random_flip_1/random_flip_left_right/control_dependency:output:0<random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА20
.random_flip_1/random_flip_left_right/ReverseV2А
(random_flip_1/random_flip_left_right/mulMul.random_flip_1/random_flip_left_right/Round:y:07random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/mulЭ
*random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2,
*random_flip_1/random_flip_left_right/sub/xъ
(random_flip_1/random_flip_left_right/subSub3random_flip_1/random_flip_left_right/sub/x:output:0.random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2*
(random_flip_1/random_flip_left_right/subЛ
*random_flip_1/random_flip_left_right/mul_1Mul,random_flip_1/random_flip_left_right/sub:z:0@random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2,
*random_flip_1/random_flip_left_right/mul_1ч
(random_flip_1/random_flip_left_right/addAddV2,random_flip_1/random_flip_left_right/mul:z:0.random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2*
(random_flip_1/random_flip_left_right/addЯ
4random_flip_1/random_flip_up_down/control_dependencyIdentity,random_flip_1/random_flip_left_right/add:z:0*
T0*;
_class1
/-loc:@random_flip_1/random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€АА26
4random_flip_1/random_flip_up_down/control_dependencyњ
'random_flip_1/random_flip_up_down/ShapeShape=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:2)
'random_flip_1/random_flip_up_down/ShapeЄ
5random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_flip_1/random_flip_up_down/strided_slice/stackЉ
7random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_1Љ
7random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_flip_1/random_flip_up_down/strided_slice/stack_2Ѓ
/random_flip_1/random_flip_up_down/strided_sliceStridedSlice0random_flip_1/random_flip_up_down/Shape:output:0>random_flip_1/random_flip_up_down/strided_slice/stack:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0@random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_flip_1/random_flip_up_down/strided_sliceа
6random_flip_1/random_flip_up_down/random_uniform/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:28
6random_flip_1/random_flip_up_down/random_uniform/shape±
4random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4random_flip_1/random_flip_up_down/random_uniform/min±
4random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4random_flip_1/random_flip_up_down/random_uniform/maxН
>random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniform?random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02@
>random_flip_1/random_flip_up_down/random_uniform/RandomUniform©
4random_flip_1/random_flip_up_down/random_uniform/MulMulGrandom_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0=random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€26
4random_flip_1/random_flip_up_down/random_uniform/Mul®
1random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/1®
1random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/2®
1random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :23
1random_flip_1/random_flip_up_down/Reshape/shape/3Ж
/random_flip_1/random_flip_up_down/Reshape/shapePack8random_flip_1/random_flip_up_down/strided_slice:output:0:random_flip_1/random_flip_up_down/Reshape/shape/1:output:0:random_flip_1/random_flip_up_down/Reshape/shape/2:output:0:random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:21
/random_flip_1/random_flip_up_down/Reshape/shapeП
)random_flip_1/random_flip_up_down/ReshapeReshape8random_flip_1/random_flip_up_down/random_uniform/Mul:z:08random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2+
)random_flip_1/random_flip_up_down/Reshape…
'random_flip_1/random_flip_up_down/RoundRound2random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2)
'random_flip_1/random_flip_up_down/RoundЃ
0random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:22
0random_flip_1/random_flip_up_down/ReverseV2/axisЭ
+random_flip_1/random_flip_up_down/ReverseV2	ReverseV2=random_flip_1/random_flip_up_down/control_dependency:output:09random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2-
+random_flip_1/random_flip_up_down/ReverseV2ф
%random_flip_1/random_flip_up_down/mulMul+random_flip_1/random_flip_up_down/Round:y:04random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/mulЧ
'random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_flip_1/random_flip_up_down/sub/xо
%random_flip_1/random_flip_up_down/subSub0random_flip_1/random_flip_up_down/sub/x:output:0+random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€2'
%random_flip_1/random_flip_up_down/sub€
'random_flip_1/random_flip_up_down/mul_1Mul)random_flip_1/random_flip_up_down/sub:z:0=random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2)
'random_flip_1/random_flip_up_down/mul_1л
%random_flip_1/random_flip_up_down/addAddV2)random_flip_1/random_flip_up_down/mul:z:0+random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2'
%random_flip_1/random_flip_up_down/addЛ
random_rotation_1/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2
random_rotation_1/ShapeШ
%random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%random_rotation_1/strided_slice/stackЬ
'random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_1Ь
'random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice/stack_2ќ
random_rotation_1/strided_sliceStridedSlice random_rotation_1/Shape:output:0.random_rotation_1/strided_slice/stack:output:00random_rotation_1/strided_slice/stack_1:output:00random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
random_rotation_1/strided_sliceЬ
'random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_1/stack†
)random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_1†
)random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_1/stack_2Ў
!random_rotation_1/strided_slice_1StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_1/stack:output:02random_rotation_1/strided_slice_1/stack_1:output:02random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_1Ф
random_rotation_1/CastCast*random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/CastЬ
'random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2)
'random_rotation_1/strided_slice_2/stack†
)random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_1†
)random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2+
)random_rotation_1/strided_slice_2/stack_2Ў
!random_rotation_1/strided_slice_2StridedSlice random_rotation_1/Shape:output:00random_rotation_1/strided_slice_2/stack:output:02random_rotation_1/strided_slice_2/stack_1:output:02random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2#
!random_rotation_1/strided_slice_2Ш
random_rotation_1/Cast_1Cast*random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
random_rotation_1/Cast_1і
(random_rotation_1/stateful_uniform/shapePack(random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:2*
(random_rotation_1/stateful_uniform/shapeХ
&random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њ2(
&random_rotation_1/stateful_uniform/minХ
&random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?2(
&random_rotation_1/stateful_uniform/maxЮ
(random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2*
(random_rotation_1/stateful_uniform/Constб
'random_rotation_1/stateful_uniform/ProdProd1random_rotation_1/stateful_uniform/shape:output:01random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/stateful_uniform/ProdШ
)random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :2+
)random_rotation_1/stateful_uniform/Cast/xј
)random_rotation_1/stateful_uniform/Cast_1Cast0random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 2+
)random_rotation_1/stateful_uniform/Cast_1≥
1random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkip:random_rotation_1_stateful_uniform_rngreadandskip_resource2random_rotation_1/stateful_uniform/Cast/x:output:0-random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:23
1random_rotation_1/stateful_uniform/RngReadAndSkipЇ
6random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 28
6random_rotation_1/stateful_uniform/strided_slice/stackЊ
8random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_1Њ
8random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice/stack_2Ї
0random_rotation_1/stateful_uniform/strided_sliceStridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0?random_rotation_1/stateful_uniform/strided_slice/stack:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Arandom_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask22
0random_rotation_1/stateful_uniform/strided_sliceѕ
*random_rotation_1/stateful_uniform/BitcastBitcast9random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type02,
*random_rotation_1/stateful_uniform/BitcastЊ
8random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2:
8random_rotation_1/stateful_uniform/strided_slice_1/stack¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_1¬
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:random_rotation_1/stateful_uniform/strided_slice_1/stack_2≤
2random_rotation_1/stateful_uniform/strided_slice_1StridedSlice9random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Arandom_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Crandom_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:24
2random_rotation_1/stateful_uniform/strided_slice_1’
,random_rotation_1/stateful_uniform/Bitcast_1Bitcast;random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02.
,random_rotation_1/stateful_uniform/Bitcast_1ƒ
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2A
?random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg§
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV21random_rotation_1/stateful_uniform/shape:output:05random_rotation_1/stateful_uniform/Bitcast_1:output:03random_rotation_1/stateful_uniform/Bitcast:output:0Hrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€2=
;random_rotation_1/stateful_uniform/StatelessRandomUniformV2Џ
&random_rotation_1/stateful_uniform/subSub/random_rotation_1/stateful_uniform/max:output:0/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 2(
&random_rotation_1/stateful_uniform/subч
&random_rotation_1/stateful_uniform/mulMulDrandom_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:0*random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€2(
&random_rotation_1/stateful_uniform/mulЏ
"random_rotation_1/stateful_uniformAdd*random_rotation_1/stateful_uniform/mul:z:0/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€2$
"random_rotation_1/stateful_uniformЧ
'random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2)
'random_rotation_1/rotation_matrix/sub/y∆
%random_rotation_1/rotation_matrix/subSubrandom_rotation_1/Cast_1:y:00random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 2'
%random_rotation_1/rotation_matrix/subЂ
%random_rotation_1/rotation_matrix/CosCos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/CosЫ
)random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_1/yћ
'random_rotation_1/rotation_matrix/sub_1Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_1џ
%random_rotation_1/rotation_matrix/mulMul)random_rotation_1/rotation_matrix/Cos:y:0+random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/mulЂ
%random_rotation_1/rotation_matrix/SinSin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/SinЫ
)random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_2/y 
'random_rotation_1/rotation_matrix/sub_2Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_2я
'random_rotation_1/rotation_matrix/mul_1Mul)random_rotation_1/rotation_matrix/Sin:y:0+random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_1я
'random_rotation_1/rotation_matrix/sub_3Sub)random_rotation_1/rotation_matrix/mul:z:0+random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_3я
'random_rotation_1/rotation_matrix/sub_4Sub)random_rotation_1/rotation_matrix/sub:z:0+random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_4Я
+random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2-
+random_rotation_1/rotation_matrix/truediv/yт
)random_rotation_1/rotation_matrix/truedivRealDiv+random_rotation_1/rotation_matrix/sub_4:z:04random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2+
)random_rotation_1/rotation_matrix/truedivЫ
)random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_5/y 
'random_rotation_1/rotation_matrix/sub_5Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_5ѓ
'random_rotation_1/rotation_matrix/Sin_1Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_1Ы
)random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_6/yћ
'random_rotation_1/rotation_matrix/sub_6Subrandom_rotation_1/Cast_1:y:02random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_6б
'random_rotation_1/rotation_matrix/mul_2Mul+random_rotation_1/rotation_matrix/Sin_1:y:0+random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_2ѓ
'random_rotation_1/rotation_matrix/Cos_1Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_1Ы
)random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2+
)random_rotation_1/rotation_matrix/sub_7/y 
'random_rotation_1/rotation_matrix/sub_7Subrandom_rotation_1/Cast:y:02random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 2)
'random_rotation_1/rotation_matrix/sub_7б
'random_rotation_1/rotation_matrix/mul_3Mul+random_rotation_1/rotation_matrix/Cos_1:y:0+random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/mul_3я
%random_rotation_1/rotation_matrix/addAddV2+random_rotation_1/rotation_matrix/mul_2:z:0+random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/addя
'random_rotation_1/rotation_matrix/sub_8Sub+random_rotation_1/rotation_matrix/sub_5:z:0)random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/sub_8£
-random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2/
-random_rotation_1/rotation_matrix/truediv_1/yш
+random_rotation_1/rotation_matrix/truediv_1RealDiv+random_rotation_1/rotation_matrix/sub_8:z:06random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2-
+random_rotation_1/rotation_matrix/truediv_1®
'random_rotation_1/rotation_matrix/ShapeShape&random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:2)
'random_rotation_1/rotation_matrix/ShapeЄ
5random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 27
5random_rotation_1/rotation_matrix/strided_slice/stackЉ
7random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_1Љ
7random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:29
7random_rotation_1/rotation_matrix/strided_slice/stack_2Ѓ
/random_rotation_1/rotation_matrix/strided_sliceStridedSlice0random_rotation_1/rotation_matrix/Shape:output:0>random_rotation_1/rotation_matrix/strided_slice/stack:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0@random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask21
/random_rotation_1/rotation_matrix/strided_sliceѓ
'random_rotation_1/rotation_matrix/Cos_2Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_2√
7random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_1/stack«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_1/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_1StridedSlice+random_rotation_1/rotation_matrix/Cos_2:y:0@random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_1ѓ
'random_rotation_1/rotation_matrix/Sin_2Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_2√
7random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_2/stack«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_2/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_2StridedSlice+random_rotation_1/rotation_matrix/Sin_2:y:0@random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_2√
%random_rotation_1/rotation_matrix/NegNeg:random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€2'
%random_rotation_1/rotation_matrix/Neg√
7random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_3/stack«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_3/stack_2е
1random_rotation_1/rotation_matrix/strided_slice_3StridedSlice-random_rotation_1/rotation_matrix/truediv:z:0@random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_3ѓ
'random_rotation_1/rotation_matrix/Sin_3Sin&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Sin_3√
7random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_4/stack«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_4/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_4StridedSlice+random_rotation_1/rotation_matrix/Sin_3:y:0@random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_4ѓ
'random_rotation_1/rotation_matrix/Cos_3Cos&random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/Cos_3√
7random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_5/stack«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_5/stack_2г
1random_rotation_1/rotation_matrix/strided_slice_5StridedSlice+random_rotation_1/rotation_matrix/Cos_3:y:0@random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_5√
7random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        29
7random_rotation_1/rotation_matrix/strided_slice_6/stack«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_1«
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2;
9random_rotation_1/rotation_matrix/strided_slice_6/stack_2з
1random_rotation_1/rotation_matrix/strided_slice_6StridedSlice/random_rotation_1/rotation_matrix/truediv_1:z:0@random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Brandom_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask23
1random_rotation_1/rotation_matrix/strided_slice_6†
-random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/zeros/mul/yф
+random_rotation_1/rotation_matrix/zeros/mulMul8random_rotation_1/rotation_matrix/strided_slice:output:06random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2-
+random_rotation_1/rotation_matrix/zeros/mul£
.random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и20
.random_rotation_1/rotation_matrix/zeros/Less/yп
,random_rotation_1/rotation_matrix/zeros/LessLess/random_rotation_1/rotation_matrix/zeros/mul:z:07random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2.
,random_rotation_1/rotation_matrix/zeros/Less¶
0random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :22
0random_rotation_1/rotation_matrix/zeros/packed/1Л
.random_rotation_1/rotation_matrix/zeros/packedPack8random_rotation_1/rotation_matrix/strided_slice:output:09random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:20
.random_rotation_1/rotation_matrix/zeros/packed£
-random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2/
-random_rotation_1/rotation_matrix/zeros/Constэ
'random_rotation_1/rotation_matrix/zerosFill7random_rotation_1/rotation_matrix/zeros/packed:output:06random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€2)
'random_rotation_1/rotation_matrix/zeros†
-random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-random_rotation_1/rotation_matrix/concat/axis№
(random_rotation_1/rotation_matrix/concatConcatV2:random_rotation_1/rotation_matrix/strided_slice_1:output:0)random_rotation_1/rotation_matrix/Neg:y:0:random_rotation_1/rotation_matrix/strided_slice_3:output:0:random_rotation_1/rotation_matrix/strided_slice_4:output:0:random_rotation_1/rotation_matrix/strided_slice_5:output:0:random_rotation_1/rotation_matrix/strided_slice_6:output:00random_rotation_1/rotation_matrix/zeros:output:06random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€2*
(random_rotation_1/rotation_matrix/concatЯ
!random_rotation_1/transform/ShapeShape)random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2#
!random_rotation_1/transform/Shapeђ
/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:21
/random_rotation_1/transform/strided_slice/stack∞
1random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_1∞
1random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1random_rotation_1/transform/strided_slice/stack_2ц
)random_rotation_1/transform/strided_sliceStridedSlice*random_rotation_1/transform/Shape:output:08random_rotation_1/transform/strided_slice/stack:output:0:random_rotation_1/transform/strided_slice/stack_1:output:0:random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2+
)random_rotation_1/transform/strided_sliceХ
&random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    2(
&random_rotation_1/transform/fill_value∆
6random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV3)random_flip_1/random_flip_up_down/add:z:01random_rotation_1/rotation_matrix/concat:output:02random_rotation_1/transform/strided_slice:output:0/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR28
6random_rotation_1/transform/ImageProjectiveTransformV3Ё
IdentityIdentityKrandom_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:02^random_rotation_1/stateful_uniform/RngReadAndSkip*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€АА: 2f
1random_rotation_1/stateful_uniform/RngReadAndSkip1random_rotation_1/stateful_uniform/RngReadAndSkip:f b
1
_output_shapes
:€€€€€€€€€АА
-
_user_specified_namerandom_flip_1_input
…
Э
(__inference_conv2d_18_layer_call_fn_7887

inputs!
unknown: 
	unknown_0: 
identityИҐStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€юю *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_18_layer_call_and_return_conditional_losses_60822
StatefulPartitionedCallШ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:€€€€€€€€€юю 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
†
ь
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7898

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp¶
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpК
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2	
BiasAddb
ReluReluBiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
Relu°
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:€€€€€€€€€юю 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:€€€€€€€€€АА: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_21_layer_call_and_return_conditional_losses_6136

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ

b
F__inference_sequential_5_layer_call_and_return_conditional_losses_7486

inputs
identityБ
resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2
resizing_1/resize/size’
 resizing_1/resize/ResizeBilinearResizeBilinearinputsresizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2"
 resizing_1/resize/ResizeBilinearm
rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2
rescaling_1/Cast/xq
rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
rescaling_1/Cast_1/xµ
rescaling_1/mulMul1resizing_1/resize/ResizeBilinear:resized_images:0rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/mulЫ
rescaling_1/addAddV2rescaling_1/mul:z:0rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
rescaling_1/addq
IdentityIdentityrescaling_1/add:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:Y U
1
_output_shapes
:€€€€€€€€€АА
 
_user_specified_nameinputs
е
_
C__inference_flatten_3_layer_call_and_return_conditional_losses_8009

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:€€€€€€€€€ :W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
™
f
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6056

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ѕ
Э
(__inference_conv2d_20_layer_call_fn_7927

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€<< *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_20_layer_call_and_return_conditional_losses_61182
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€<< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>> : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€>> 
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_20_layer_call_and_return_conditional_losses_6118

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€<< 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€>> : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€>> 
 
_user_specified_nameinputs
Ж
Q
+__inference_sequential_5_layer_call_fn_7451
resizing_1_input
identityЎ
PartitionedCallPartitionedCallresizing_1_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:€€€€€€€€€АА* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *O
fJRH
F__inference_sequential_5_layer_call_and_return_conditional_losses_55682
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:€€€€€€€€€АА:c _
1
_output_shapes
:€€€€€€€€€АА
*
_user_specified_nameresizing_1_input
’
K
/__inference_max_pooling2d_22_layer_call_fn_6050

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_60442
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
’
K
/__inference_max_pooling2d_23_layer_call_fn_6062

inputs
identityл
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *S
fNRL
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_60562
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ф
ь
C__inference_conv2d_21_layer_call_and_return_conditional_losses_7958

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ѕ
Э
(__inference_conv2d_22_layer_call_fn_7967

inputs!
unknown:  
	unknown_0: 
identityИҐStatefulPartitionedCallы
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€ *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *L
fGRE
C__inference_conv2d_22_layer_call_and_return_conditional_losses_61542
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs
Ш
Ф
&__inference_dense_6_layer_call_fn_8018

inputs
unknown:	А@
	unknown_0:@
identityИҐStatefulPartitionedCallс
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *J
fERC
A__inference_dense_6_layer_call_and_return_conditional_losses_61982
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:€€€€€€€€€А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А
 
_user_specified_nameinputs
”«
ъ
F__inference_sequential_7_layer_call_and_return_conditional_losses_7446
sequential_5_inputU
Gsequential_6_random_rotation_1_stateful_uniform_rngreadandskip_resource:	B
(conv2d_18_conv2d_readvariableop_resource: 7
)conv2d_18_biasadd_readvariableop_resource: B
(conv2d_19_conv2d_readvariableop_resource:  7
)conv2d_19_biasadd_readvariableop_resource: B
(conv2d_20_conv2d_readvariableop_resource:  7
)conv2d_20_biasadd_readvariableop_resource: B
(conv2d_21_conv2d_readvariableop_resource:  7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:  7
)conv2d_22_biasadd_readvariableop_resource: B
(conv2d_23_conv2d_readvariableop_resource:  7
)conv2d_23_biasadd_readvariableop_resource: 9
&dense_6_matmul_readvariableop_resource:	А@5
'dense_6_biasadd_readvariableop_resource:@8
&dense_7_matmul_readvariableop_resource:@5
'dense_7_biasadd_readvariableop_resource:
identityИҐ conv2d_18/BiasAdd/ReadVariableOpҐconv2d_18/Conv2D/ReadVariableOpҐ conv2d_19/BiasAdd/ReadVariableOpҐconv2d_19/Conv2D/ReadVariableOpҐ conv2d_20/BiasAdd/ReadVariableOpҐconv2d_20/Conv2D/ReadVariableOpҐ conv2d_21/BiasAdd/ReadVariableOpҐconv2d_21/Conv2D/ReadVariableOpҐ conv2d_22/BiasAdd/ReadVariableOpҐconv2d_22/Conv2D/ReadVariableOpҐ conv2d_23/BiasAdd/ReadVariableOpҐconv2d_23/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOpҐdense_7/BiasAdd/ReadVariableOpҐdense_7/MatMul/ReadVariableOpҐ>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkipЫ
#sequential_5/resizing_1/resize/sizeConst*
_output_shapes
:*
dtype0*
valueB"      2%
#sequential_5/resizing_1/resize/sizeИ
-sequential_5/resizing_1/resize/ResizeBilinearResizeBilinearsequential_5_input,sequential_5/resizing_1/resize/size:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА*
half_pixel_centers(2/
-sequential_5/resizing_1/resize/ResizeBilinearЗ
sequential_5/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *БАА;2!
sequential_5/rescaling_1/Cast/xЛ
!sequential_5/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!sequential_5/rescaling_1/Cast_1/xй
sequential_5/rescaling_1/mulMul>sequential_5/resizing_1/resize/ResizeBilinear:resized_images:0(sequential_5/rescaling_1/Cast/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/mulѕ
sequential_5/rescaling_1/addAddV2 sequential_5/rescaling_1/mul:z:0*sequential_5/rescaling_1/Cast_1/x:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2
sequential_5/rescaling_1/addІ
Dsequential_6/random_flip_1/random_flip_left_right/control_dependencyIdentity sequential_5/rescaling_1/add:z:0*
T0*/
_class%
#!loc:@sequential_5/rescaling_1/add*1
_output_shapes
:€€€€€€€€€АА2F
Dsequential_6/random_flip_1/random_flip_left_right/control_dependencyп
7sequential_6/random_flip_1/random_flip_left_right/ShapeShapeMsequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*
_output_shapes
:29
7sequential_6/random_flip_1/random_flip_left_right/ShapeЎ
Esequential_6/random_flip_1/random_flip_left_right/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2G
Esequential_6/random_flip_1/random_flip_left_right/strided_slice/stack№
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1№
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2О
?sequential_6/random_flip_1/random_flip_left_right/strided_sliceStridedSlice@sequential_6/random_flip_1/random_flip_left_right/Shape:output:0Nsequential_6/random_flip_1/random_flip_left_right/strided_slice/stack:output:0Psequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_1:output:0Psequential_6/random_flip_1/random_flip_left_right/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2A
?sequential_6/random_flip_1/random_flip_left_right/strided_sliceР
Fsequential_6/random_flip_1/random_flip_left_right/random_uniform/shapePackHsequential_6/random_flip_1/random_flip_left_right/strided_slice:output:0*
N*
T0*
_output_shapes
:2H
Fsequential_6/random_flip_1/random_flip_left_right/random_uniform/shape—
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/min—
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/maxљ
Nsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniformRandomUniformOsequential_6/random_flip_1/random_flip_left_right/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02P
Nsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniformй
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/MulMulWsequential_6/random_flip_1/random_flip_left_right/random_uniform/RandomUniform:output:0Msequential_6/random_flip_1/random_flip_left_right/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€2F
Dsequential_6/random_flip_1/random_flip_left_right/random_uniform/Mul»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2»
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2C
Asequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3ж
?sequential_6/random_flip_1/random_flip_left_right/Reshape/shapePackHsequential_6/random_flip_1/random_flip_left_right/strided_slice:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/1:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/2:output:0Jsequential_6/random_flip_1/random_flip_left_right/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2A
?sequential_6/random_flip_1/random_flip_left_right/Reshape/shapeѕ
9sequential_6/random_flip_1/random_flip_left_right/ReshapeReshapeHsequential_6/random_flip_1/random_flip_left_right/random_uniform/Mul:z:0Hsequential_6/random_flip_1/random_flip_left_right/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€2;
9sequential_6/random_flip_1/random_flip_left_right/Reshapeщ
7sequential_6/random_flip_1/random_flip_left_right/RoundRoundBsequential_6/random_flip_1/random_flip_left_right/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€29
7sequential_6/random_flip_1/random_flip_left_right/Roundќ
@sequential_6/random_flip_1/random_flip_left_right/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2B
@sequential_6/random_flip_1/random_flip_left_right/ReverseV2/axisЁ
;sequential_6/random_flip_1/random_flip_left_right/ReverseV2	ReverseV2Msequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0Isequential_6/random_flip_1/random_flip_left_right/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2=
;sequential_6/random_flip_1/random_flip_left_right/ReverseV2і
5sequential_6/random_flip_1/random_flip_left_right/mulMul;sequential_6/random_flip_1/random_flip_left_right/Round:y:0Dsequential_6/random_flip_1/random_flip_left_right/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА27
5sequential_6/random_flip_1/random_flip_left_right/mulЈ
7sequential_6/random_flip_1/random_flip_left_right/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?29
7sequential_6/random_flip_1/random_flip_left_right/sub/xЃ
5sequential_6/random_flip_1/random_flip_left_right/subSub@sequential_6/random_flip_1/random_flip_left_right/sub/x:output:0;sequential_6/random_flip_1/random_flip_left_right/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€27
5sequential_6/random_flip_1/random_flip_left_right/subњ
7sequential_6/random_flip_1/random_flip_left_right/mul_1Mul9sequential_6/random_flip_1/random_flip_left_right/sub:z:0Msequential_6/random_flip_1/random_flip_left_right/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА29
7sequential_6/random_flip_1/random_flip_left_right/mul_1Ђ
5sequential_6/random_flip_1/random_flip_left_right/addAddV29sequential_6/random_flip_1/random_flip_left_right/mul:z:0;sequential_6/random_flip_1/random_flip_left_right/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА27
5sequential_6/random_flip_1/random_flip_left_right/add”
Asequential_6/random_flip_1/random_flip_up_down/control_dependencyIdentity9sequential_6/random_flip_1/random_flip_left_right/add:z:0*
T0*H
_class>
<:loc:@sequential_6/random_flip_1/random_flip_left_right/add*1
_output_shapes
:€€€€€€€€€АА2C
Asequential_6/random_flip_1/random_flip_up_down/control_dependencyж
4sequential_6/random_flip_1/random_flip_up_down/ShapeShapeJsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*
_output_shapes
:26
4sequential_6/random_flip_1/random_flip_up_down/Shape“
Bsequential_6/random_flip_1/random_flip_up_down/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack÷
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1÷
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2ь
<sequential_6/random_flip_1/random_flip_up_down/strided_sliceStridedSlice=sequential_6/random_flip_1/random_flip_up_down/Shape:output:0Ksequential_6/random_flip_1/random_flip_up_down/strided_slice/stack:output:0Msequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_1:output:0Msequential_6/random_flip_1/random_flip_up_down/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/random_flip_1/random_flip_up_down/strided_sliceЗ
Csequential_6/random_flip_1/random_flip_up_down/random_uniform/shapePackEsequential_6/random_flip_1/random_flip_up_down/strided_slice:output:0*
N*
T0*
_output_shapes
:2E
Csequential_6/random_flip_1/random_flip_up_down/random_uniform/shapeЋ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *    2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/minЋ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *  А?2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/maxі
Ksequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniformRandomUniformLsequential_6/random_flip_1/random_flip_up_down/random_uniform/shape:output:0*
T0*#
_output_shapes
:€€€€€€€€€*
dtype02M
Ksequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniformЁ
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/MulMulTsequential_6/random_flip_1/random_flip_up_down/random_uniform/RandomUniform:output:0Jsequential_6/random_flip_1/random_flip_up_down/random_uniform/max:output:0*
T0*#
_output_shapes
:€€€€€€€€€2C
Asequential_6/random_flip_1/random_flip_up_down/random_uniform/Mul¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2¬
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2@
>sequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3‘
<sequential_6/random_flip_1/random_flip_up_down/Reshape/shapePackEsequential_6/random_flip_1/random_flip_up_down/strided_slice:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/1:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/2:output:0Gsequential_6/random_flip_1/random_flip_up_down/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2>
<sequential_6/random_flip_1/random_flip_up_down/Reshape/shape√
6sequential_6/random_flip_1/random_flip_up_down/ReshapeReshapeEsequential_6/random_flip_1/random_flip_up_down/random_uniform/Mul:z:0Esequential_6/random_flip_1/random_flip_up_down/Reshape/shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€28
6sequential_6/random_flip_1/random_flip_up_down/Reshapeр
4sequential_6/random_flip_1/random_flip_up_down/RoundRound?sequential_6/random_flip_1/random_flip_up_down/Reshape:output:0*
T0*/
_output_shapes
:€€€€€€€€€26
4sequential_6/random_flip_1/random_flip_up_down/Round»
=sequential_6/random_flip_1/random_flip_up_down/ReverseV2/axisConst*
_output_shapes
:*
dtype0*
valueB:2?
=sequential_6/random_flip_1/random_flip_up_down/ReverseV2/axis—
8sequential_6/random_flip_1/random_flip_up_down/ReverseV2	ReverseV2Jsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0Fsequential_6/random_flip_1/random_flip_up_down/ReverseV2/axis:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА2:
8sequential_6/random_flip_1/random_flip_up_down/ReverseV2®
2sequential_6/random_flip_1/random_flip_up_down/mulMul8sequential_6/random_flip_1/random_flip_up_down/Round:y:0Asequential_6/random_flip_1/random_flip_up_down/ReverseV2:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА24
2sequential_6/random_flip_1/random_flip_up_down/mul±
4sequential_6/random_flip_1/random_flip_up_down/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4sequential_6/random_flip_1/random_flip_up_down/sub/xҐ
2sequential_6/random_flip_1/random_flip_up_down/subSub=sequential_6/random_flip_1/random_flip_up_down/sub/x:output:08sequential_6/random_flip_1/random_flip_up_down/Round:y:0*
T0*/
_output_shapes
:€€€€€€€€€24
2sequential_6/random_flip_1/random_flip_up_down/sub≥
4sequential_6/random_flip_1/random_flip_up_down/mul_1Mul6sequential_6/random_flip_1/random_flip_up_down/sub:z:0Jsequential_6/random_flip_1/random_flip_up_down/control_dependency:output:0*
T0*1
_output_shapes
:€€€€€€€€€АА26
4sequential_6/random_flip_1/random_flip_up_down/mul_1Я
2sequential_6/random_flip_1/random_flip_up_down/addAddV26sequential_6/random_flip_1/random_flip_up_down/mul:z:08sequential_6/random_flip_1/random_flip_up_down/mul_1:z:0*
T0*1
_output_shapes
:€€€€€€€€€АА24
2sequential_6/random_flip_1/random_flip_up_down/add≤
$sequential_6/random_rotation_1/ShapeShape6sequential_6/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:2&
$sequential_6/random_rotation_1/Shape≤
2sequential_6/random_rotation_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 24
2sequential_6/random_rotation_1/strided_slice/stackґ
4sequential_6/random_rotation_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice/stack_1ґ
4sequential_6/random_rotation_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice/stack_2Ь
,sequential_6/random_rotation_1/strided_sliceStridedSlice-sequential_6/random_rotation_1/Shape:output:0;sequential_6/random_rotation_1/strided_slice/stack:output:0=sequential_6/random_rotation_1/strided_slice/stack_1:output:0=sequential_6/random_rotation_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2.
,sequential_6/random_rotation_1/strided_sliceґ
4sequential_6/random_rotation_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice_1/stackЇ
6sequential_6/random_rotation_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_1/stack_1Ї
6sequential_6/random_rotation_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_1/stack_2¶
.sequential_6/random_rotation_1/strided_slice_1StridedSlice-sequential_6/random_rotation_1/Shape:output:0=sequential_6/random_rotation_1/strided_slice_1/stack:output:0?sequential_6/random_rotation_1/strided_slice_1/stack_1:output:0?sequential_6/random_rotation_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_6/random_rotation_1/strided_slice_1ї
#sequential_6/random_rotation_1/CastCast7sequential_6/random_rotation_1/strided_slice_1:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#sequential_6/random_rotation_1/Castґ
4sequential_6/random_rotation_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:26
4sequential_6/random_rotation_1/strided_slice_2/stackЇ
6sequential_6/random_rotation_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_2/stack_1Ї
6sequential_6/random_rotation_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6sequential_6/random_rotation_1/strided_slice_2/stack_2¶
.sequential_6/random_rotation_1/strided_slice_2StridedSlice-sequential_6/random_rotation_1/Shape:output:0=sequential_6/random_rotation_1/strided_slice_2/stack:output:0?sequential_6/random_rotation_1/strided_slice_2/stack_1:output:0?sequential_6/random_rotation_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.sequential_6/random_rotation_1/strided_slice_2њ
%sequential_6/random_rotation_1/Cast_1Cast7sequential_6/random_rotation_1/strided_slice_2:output:0*

DstT0*

SrcT0*
_output_shapes
: 2'
%sequential_6/random_rotation_1/Cast_1џ
5sequential_6/random_rotation_1/stateful_uniform/shapePack5sequential_6/random_rotation_1/strided_slice:output:0*
N*
T0*
_output_shapes
:27
5sequential_6/random_rotation_1/stateful_uniform/shapeѓ
3sequential_6/random_rotation_1/stateful_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†њ25
3sequential_6/random_rotation_1/stateful_uniform/minѓ
3sequential_6/random_rotation_1/stateful_uniform/maxConst*
_output_shapes
: *
dtype0*
valueB
 *|ў†?25
3sequential_6/random_rotation_1/stateful_uniform/maxЄ
5sequential_6/random_rotation_1/stateful_uniform/ConstConst*
_output_shapes
:*
dtype0*
valueB: 27
5sequential_6/random_rotation_1/stateful_uniform/ConstХ
4sequential_6/random_rotation_1/stateful_uniform/ProdProd>sequential_6/random_rotation_1/stateful_uniform/shape:output:0>sequential_6/random_rotation_1/stateful_uniform/Const:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/stateful_uniform/Prod≤
6sequential_6/random_rotation_1/stateful_uniform/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :28
6sequential_6/random_rotation_1/stateful_uniform/Cast/xз
6sequential_6/random_rotation_1/stateful_uniform/Cast_1Cast=sequential_6/random_rotation_1/stateful_uniform/Prod:output:0*

DstT0*

SrcT0*
_output_shapes
: 28
6sequential_6/random_rotation_1/stateful_uniform/Cast_1ф
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkipRngReadAndSkipGsequential_6_random_rotation_1_stateful_uniform_rngreadandskip_resource?sequential_6/random_rotation_1/stateful_uniform/Cast/x:output:0:sequential_6/random_rotation_1/stateful_uniform/Cast_1:y:0*
_output_shapes
:2@
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip‘
Csequential_6/random_rotation_1/stateful_uniform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2E
Csequential_6/random_rotation_1/stateful_uniform/strided_slice/stackЎ
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1Ў
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2И
=sequential_6/random_rotation_1/stateful_uniform/strided_sliceStridedSliceFsequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Lsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack:output:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_1:output:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice/stack_2:output:0*
Index0*
T0	*
_output_shapes
:*

begin_mask2?
=sequential_6/random_rotation_1/stateful_uniform/strided_sliceц
7sequential_6/random_rotation_1/stateful_uniform/BitcastBitcastFsequential_6/random_rotation_1/stateful_uniform/strided_slice:output:0*
T0	*
_output_shapes
:*

type029
7sequential_6/random_rotation_1/stateful_uniform/BitcastЎ
Esequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2G
Esequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack№
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1№
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2I
Gsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2А
?sequential_6/random_rotation_1/stateful_uniform/strided_slice_1StridedSliceFsequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:value:0Nsequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack:output:0Psequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_1:output:0Psequential_6/random_rotation_1/stateful_uniform/strided_slice_1/stack_2:output:0*
Index0*
T0	*
_output_shapes
:2A
?sequential_6/random_rotation_1/stateful_uniform/strided_slice_1ь
9sequential_6/random_rotation_1/stateful_uniform/Bitcast_1BitcastHsequential_6/random_rotation_1/stateful_uniform/strided_slice_1:output:0*
T0	*
_output_shapes
:*

type02;
9sequential_6/random_rotation_1/stateful_uniform/Bitcast_1ё
Lsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algConst*
_output_shapes
: *
dtype0*
value	B :2N
Lsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/algт
Hsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2StatelessRandomUniformV2>sequential_6/random_rotation_1/stateful_uniform/shape:output:0Bsequential_6/random_rotation_1/stateful_uniform/Bitcast_1:output:0@sequential_6/random_rotation_1/stateful_uniform/Bitcast:output:0Usequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2/alg:output:0*#
_output_shapes
:€€€€€€€€€2J
Hsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2О
3sequential_6/random_rotation_1/stateful_uniform/subSub<sequential_6/random_rotation_1/stateful_uniform/max:output:0<sequential_6/random_rotation_1/stateful_uniform/min:output:0*
T0*
_output_shapes
: 25
3sequential_6/random_rotation_1/stateful_uniform/subЂ
3sequential_6/random_rotation_1/stateful_uniform/mulMulQsequential_6/random_rotation_1/stateful_uniform/StatelessRandomUniformV2:output:07sequential_6/random_rotation_1/stateful_uniform/sub:z:0*
T0*#
_output_shapes
:€€€€€€€€€25
3sequential_6/random_rotation_1/stateful_uniform/mulО
/sequential_6/random_rotation_1/stateful_uniformAdd7sequential_6/random_rotation_1/stateful_uniform/mul:z:0<sequential_6/random_rotation_1/stateful_uniform/min:output:0*
T0*#
_output_shapes
:€€€€€€€€€21
/sequential_6/random_rotation_1/stateful_uniform±
4sequential_6/random_rotation_1/rotation_matrix/sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?26
4sequential_6/random_rotation_1/rotation_matrix/sub/yъ
2sequential_6/random_rotation_1/rotation_matrix/subSub)sequential_6/random_rotation_1/Cast_1:y:0=sequential_6/random_rotation_1/rotation_matrix/sub/y:output:0*
T0*
_output_shapes
: 24
2sequential_6/random_rotation_1/rotation_matrix/sub“
2sequential_6/random_rotation_1/rotation_matrix/CosCos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/Cosµ
6sequential_6/random_rotation_1/rotation_matrix/sub_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_1/yА
4sequential_6/random_rotation_1/rotation_matrix/sub_1Sub)sequential_6/random_rotation_1/Cast_1:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_1/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_1П
2sequential_6/random_rotation_1/rotation_matrix/mulMul6sequential_6/random_rotation_1/rotation_matrix/Cos:y:08sequential_6/random_rotation_1/rotation_matrix/sub_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/mul“
2sequential_6/random_rotation_1/rotation_matrix/SinSin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/Sinµ
6sequential_6/random_rotation_1/rotation_matrix/sub_2/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_2/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_2Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_2/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_2У
4sequential_6/random_rotation_1/rotation_matrix/mul_1Mul6sequential_6/random_rotation_1/rotation_matrix/Sin:y:08sequential_6/random_rotation_1/rotation_matrix/sub_2:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_1У
4sequential_6/random_rotation_1/rotation_matrix/sub_3Sub6sequential_6/random_rotation_1/rotation_matrix/mul:z:08sequential_6/random_rotation_1/rotation_matrix/mul_1:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_3У
4sequential_6/random_rotation_1/rotation_matrix/sub_4Sub6sequential_6/random_rotation_1/rotation_matrix/sub:z:08sequential_6/random_rotation_1/rotation_matrix/sub_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_4є
8sequential_6/random_rotation_1/rotation_matrix/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2:
8sequential_6/random_rotation_1/rotation_matrix/truediv/y¶
6sequential_6/random_rotation_1/rotation_matrix/truedivRealDiv8sequential_6/random_rotation_1/rotation_matrix/sub_4:z:0Asequential_6/random_rotation_1/rotation_matrix/truediv/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€28
6sequential_6/random_rotation_1/rotation_matrix/truedivµ
6sequential_6/random_rotation_1/rotation_matrix/sub_5/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_5/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_5Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_5/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_5÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_1Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_1µ
6sequential_6/random_rotation_1/rotation_matrix/sub_6/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_6/yА
4sequential_6/random_rotation_1/rotation_matrix/sub_6Sub)sequential_6/random_rotation_1/Cast_1:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_6/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_6Х
4sequential_6/random_rotation_1/rotation_matrix/mul_2Mul8sequential_6/random_rotation_1/rotation_matrix/Sin_1:y:08sequential_6/random_rotation_1/rotation_matrix/sub_6:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_2÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_1Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_1µ
6sequential_6/random_rotation_1/rotation_matrix/sub_7/yConst*
_output_shapes
: *
dtype0*
valueB
 *  А?28
6sequential_6/random_rotation_1/rotation_matrix/sub_7/yю
4sequential_6/random_rotation_1/rotation_matrix/sub_7Sub'sequential_6/random_rotation_1/Cast:y:0?sequential_6/random_rotation_1/rotation_matrix/sub_7/y:output:0*
T0*
_output_shapes
: 26
4sequential_6/random_rotation_1/rotation_matrix/sub_7Х
4sequential_6/random_rotation_1/rotation_matrix/mul_3Mul8sequential_6/random_rotation_1/rotation_matrix/Cos_1:y:08sequential_6/random_rotation_1/rotation_matrix/sub_7:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/mul_3У
2sequential_6/random_rotation_1/rotation_matrix/addAddV28sequential_6/random_rotation_1/rotation_matrix/mul_2:z:08sequential_6/random_rotation_1/rotation_matrix/mul_3:z:0*
T0*#
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/addУ
4sequential_6/random_rotation_1/rotation_matrix/sub_8Sub8sequential_6/random_rotation_1/rotation_matrix/sub_5:z:06sequential_6/random_rotation_1/rotation_matrix/add:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/sub_8љ
:sequential_6/random_rotation_1/rotation_matrix/truediv_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2<
:sequential_6/random_rotation_1/rotation_matrix/truediv_1/yђ
8sequential_6/random_rotation_1/rotation_matrix/truediv_1RealDiv8sequential_6/random_rotation_1/rotation_matrix/sub_8:z:0Csequential_6/random_rotation_1/rotation_matrix/truediv_1/y:output:0*
T0*#
_output_shapes
:€€€€€€€€€2:
8sequential_6/random_rotation_1/rotation_matrix/truediv_1ѕ
4sequential_6/random_rotation_1/rotation_matrix/ShapeShape3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*
_output_shapes
:26
4sequential_6/random_rotation_1/rotation_matrix/Shape“
Bsequential_6/random_rotation_1/rotation_matrix/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2D
Bsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack÷
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1÷
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2ь
<sequential_6/random_rotation_1/rotation_matrix/strided_sliceStridedSlice=sequential_6/random_rotation_1/rotation_matrix/Shape:output:0Ksequential_6/random_rotation_1/rotation_matrix/strided_slice/stack:output:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_1:output:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2>
<sequential_6/random_rotation_1/rotation_matrix/strided_slice÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_2Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_2Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_1StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Cos_2:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_1/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_1÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_2Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_2Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_2StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Sin_2:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_2/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_2к
2sequential_6/random_rotation_1/rotation_matrix/NegNegGsequential_6/random_rotation_1/rotation_matrix/strided_slice_2:output:0*
T0*'
_output_shapes
:€€€€€€€€€24
2sequential_6/random_rotation_1/rotation_matrix/NegЁ
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2≥
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_3StridedSlice:sequential_6/random_rotation_1/rotation_matrix/truediv:z:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_3/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_3÷
4sequential_6/random_rotation_1/rotation_matrix/Sin_3Sin3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Sin_3Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_4StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Sin_3:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_4/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_4÷
4sequential_6/random_rotation_1/rotation_matrix/Cos_3Cos3sequential_6/random_rotation_1/stateful_uniform:z:0*
T0*#
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/Cos_3Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2±
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_5StridedSlice8sequential_6/random_rotation_1/rotation_matrix/Cos_3:y:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_5/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_5Ё
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stackConst*
_output_shapes
:*
dtype0*
valueB"        2F
Dsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stackб
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1б
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2H
Fsequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2µ
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_6StridedSlice<sequential_6/random_rotation_1/rotation_matrix/truediv_1:z:0Msequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_1:output:0Osequential_6/random_rotation_1/rotation_matrix/strided_slice_6/stack_2:output:0*
Index0*
T0*'
_output_shapes
:€€€€€€€€€*

begin_mask*
end_mask*
new_axis_mask2@
>sequential_6/random_rotation_1/rotation_matrix/strided_slice_6Ї
:sequential_6/random_rotation_1/rotation_matrix/zeros/mul/yConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_6/random_rotation_1/rotation_matrix/zeros/mul/y®
8sequential_6/random_rotation_1/rotation_matrix/zeros/mulMulEsequential_6/random_rotation_1/rotation_matrix/strided_slice:output:0Csequential_6/random_rotation_1/rotation_matrix/zeros/mul/y:output:0*
T0*
_output_shapes
: 2:
8sequential_6/random_rotation_1/rotation_matrix/zeros/mulљ
;sequential_6/random_rotation_1/rotation_matrix/zeros/Less/yConst*
_output_shapes
: *
dtype0*
value
B :и2=
;sequential_6/random_rotation_1/rotation_matrix/zeros/Less/y£
9sequential_6/random_rotation_1/rotation_matrix/zeros/LessLess<sequential_6/random_rotation_1/rotation_matrix/zeros/mul:z:0Dsequential_6/random_rotation_1/rotation_matrix/zeros/Less/y:output:0*
T0*
_output_shapes
: 2;
9sequential_6/random_rotation_1/rotation_matrix/zeros/Lessј
=sequential_6/random_rotation_1/rotation_matrix/zeros/packed/1Const*
_output_shapes
: *
dtype0*
value	B :2?
=sequential_6/random_rotation_1/rotation_matrix/zeros/packed/1њ
;sequential_6/random_rotation_1/rotation_matrix/zeros/packedPackEsequential_6/random_rotation_1/rotation_matrix/strided_slice:output:0Fsequential_6/random_rotation_1/rotation_matrix/zeros/packed/1:output:0*
N*
T0*
_output_shapes
:2=
;sequential_6/random_rotation_1/rotation_matrix/zeros/packedљ
:sequential_6/random_rotation_1/rotation_matrix/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2<
:sequential_6/random_rotation_1/rotation_matrix/zeros/Const±
4sequential_6/random_rotation_1/rotation_matrix/zerosFillDsequential_6/random_rotation_1/rotation_matrix/zeros/packed:output:0Csequential_6/random_rotation_1/rotation_matrix/zeros/Const:output:0*
T0*'
_output_shapes
:€€€€€€€€€26
4sequential_6/random_rotation_1/rotation_matrix/zerosЇ
:sequential_6/random_rotation_1/rotation_matrix/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2<
:sequential_6/random_rotation_1/rotation_matrix/concat/axisё
5sequential_6/random_rotation_1/rotation_matrix/concatConcatV2Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_1:output:06sequential_6/random_rotation_1/rotation_matrix/Neg:y:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_3:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_4:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_5:output:0Gsequential_6/random_rotation_1/rotation_matrix/strided_slice_6:output:0=sequential_6/random_rotation_1/rotation_matrix/zeros:output:0Csequential_6/random_rotation_1/rotation_matrix/concat/axis:output:0*
N*
T0*'
_output_shapes
:€€€€€€€€€27
5sequential_6/random_rotation_1/rotation_matrix/concat∆
.sequential_6/random_rotation_1/transform/ShapeShape6sequential_6/random_flip_1/random_flip_up_down/add:z:0*
T0*
_output_shapes
:20
.sequential_6/random_rotation_1/transform/Shape∆
<sequential_6/random_rotation_1/transform/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2>
<sequential_6/random_rotation_1/transform/strided_slice/stack 
>sequential_6/random_rotation_1/transform/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/random_rotation_1/transform/strided_slice/stack_1 
>sequential_6/random_rotation_1/transform/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>sequential_6/random_rotation_1/transform/strided_slice/stack_2ƒ
6sequential_6/random_rotation_1/transform/strided_sliceStridedSlice7sequential_6/random_rotation_1/transform/Shape:output:0Esequential_6/random_rotation_1/transform/strided_slice/stack:output:0Gsequential_6/random_rotation_1/transform/strided_slice/stack_1:output:0Gsequential_6/random_rotation_1/transform/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:28
6sequential_6/random_rotation_1/transform/strided_sliceѓ
3sequential_6/random_rotation_1/transform/fill_valueConst*
_output_shapes
: *
dtype0*
valueB
 *    25
3sequential_6/random_rotation_1/transform/fill_valueФ
Csequential_6/random_rotation_1/transform/ImageProjectiveTransformV3ImageProjectiveTransformV36sequential_6/random_flip_1/random_flip_up_down/add:z:0>sequential_6/random_rotation_1/rotation_matrix/concat:output:0?sequential_6/random_rotation_1/transform/strided_slice:output:0<sequential_6/random_rotation_1/transform/fill_value:output:0*1
_output_shapes
:€€€€€€€€€АА*
dtype0*
	fill_mode	REFLECT*
interpolation
BILINEAR2E
Csequential_6/random_rotation_1/transform/ImageProjectiveTransformV3≥
conv2d_18/Conv2D/ReadVariableOpReadVariableOp(conv2d_18_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_18/Conv2D/ReadVariableOpЦ
conv2d_18/Conv2DConv2DXsequential_6/random_rotation_1/transform/ImageProjectiveTransformV3:transformed_images:0'conv2d_18/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю *
paddingVALID*
strides
2
conv2d_18/Conv2D™
 conv2d_18/BiasAdd/ReadVariableOpReadVariableOp)conv2d_18_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_18/BiasAdd/ReadVariableOp≤
conv2d_18/BiasAddBiasAddconv2d_18/Conv2D:output:0(conv2d_18/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/BiasAddА
conv2d_18/ReluReluconv2d_18/BiasAdd:output:0*
T0*1
_output_shapes
:€€€€€€€€€юю 2
conv2d_18/Relu 
max_pooling2d_18/MaxPoolMaxPoolconv2d_18/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_18/MaxPool≥
conv2d_19/Conv2D/ReadVariableOpReadVariableOp(conv2d_19_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_19/Conv2D/ReadVariableOpЁ
conv2d_19/Conv2DConv2D!max_pooling2d_18/MaxPool:output:0'conv2d_19/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} *
paddingVALID*
strides
2
conv2d_19/Conv2D™
 conv2d_19/BiasAdd/ReadVariableOpReadVariableOp)conv2d_19_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_19/BiasAdd/ReadVariableOp∞
conv2d_19/BiasAddBiasAddconv2d_19/Conv2D:output:0(conv2d_19/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/BiasAdd~
conv2d_19/ReluReluconv2d_19/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€}} 2
conv2d_19/Relu 
max_pooling2d_19/MaxPoolMaxPoolconv2d_19/Relu:activations:0*/
_output_shapes
:€€€€€€€€€>> *
ksize
*
paddingVALID*
strides
2
max_pooling2d_19/MaxPool≥
conv2d_20/Conv2D/ReadVariableOpReadVariableOp(conv2d_20_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_20/Conv2D/ReadVariableOpЁ
conv2d_20/Conv2DConv2D!max_pooling2d_19/MaxPool:output:0'conv2d_20/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< *
paddingVALID*
strides
2
conv2d_20/Conv2D™
 conv2d_20/BiasAdd/ReadVariableOpReadVariableOp)conv2d_20_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_20/BiasAdd/ReadVariableOp∞
conv2d_20/BiasAddBiasAddconv2d_20/Conv2D:output:0(conv2d_20/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/BiasAdd~
conv2d_20/ReluReluconv2d_20/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€<< 2
conv2d_20/Relu 
max_pooling2d_20/MaxPoolMaxPoolconv2d_20/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_20/MaxPool≥
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_21/Conv2D/ReadVariableOpЁ
conv2d_21/Conv2DConv2D!max_pooling2d_20/MaxPool:output:0'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_21/Conv2D™
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp∞
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/BiasAdd~
conv2d_21/ReluReluconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_21/Relu 
max_pooling2d_21/MaxPoolMaxPoolconv2d_21/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_21/MaxPool≥
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_22/Conv2D/ReadVariableOpЁ
conv2d_22/Conv2DConv2D!max_pooling2d_21/MaxPool:output:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_22/Conv2D™
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp∞
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/BiasAdd~
conv2d_22/ReluReluconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_22/Relu 
max_pooling2d_22/MaxPoolMaxPoolconv2d_22/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_22/MaxPool≥
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02!
conv2d_23/Conv2D/ReadVariableOpЁ
conv2d_23/Conv2DConv2D!max_pooling2d_22/MaxPool:output:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
conv2d_23/Conv2D™
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp∞
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/BiasAdd~
conv2d_23/ReluReluconv2d_23/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
conv2d_23/Relu 
max_pooling2d_23/MaxPoolMaxPoolconv2d_23/Relu:activations:0*/
_output_shapes
:€€€€€€€€€ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_23/MaxPools
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А   2
flatten_3/Const°
flatten_3/ReshapeReshape!max_pooling2d_23/MaxPool:output:0flatten_3/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А2
flatten_3/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А@*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_3/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/BiasAddp
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€@2
dense_6/Relu•
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource*
_output_shapes

:@*
dtype02
dense_7/MatMul/ReadVariableOpЯ
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/MatMul§
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp°
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/BiasAddy
dense_7/SoftmaxSoftmaxdense_7/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
dense_7/Softmaxќ
IdentityIdentitydense_7/Softmax:softmax:0!^conv2d_18/BiasAdd/ReadVariableOp ^conv2d_18/Conv2D/ReadVariableOp!^conv2d_19/BiasAdd/ReadVariableOp ^conv2d_19/Conv2D/ReadVariableOp!^conv2d_20/BiasAdd/ReadVariableOp ^conv2d_20/Conv2D/ReadVariableOp!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp^dense_7/BiasAdd/ReadVariableOp^dense_7/MatMul/ReadVariableOp?^sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*R
_input_shapesA
?:€€€€€€€€€АА: : : : : : : : : : : : : : : : : 2D
 conv2d_18/BiasAdd/ReadVariableOp conv2d_18/BiasAdd/ReadVariableOp2B
conv2d_18/Conv2D/ReadVariableOpconv2d_18/Conv2D/ReadVariableOp2D
 conv2d_19/BiasAdd/ReadVariableOp conv2d_19/BiasAdd/ReadVariableOp2B
conv2d_19/Conv2D/ReadVariableOpconv2d_19/Conv2D/ReadVariableOp2D
 conv2d_20/BiasAdd/ReadVariableOp conv2d_20/BiasAdd/ReadVariableOp2B
conv2d_20/Conv2D/ReadVariableOpconv2d_20/Conv2D/ReadVariableOp2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp2@
dense_7/BiasAdd/ReadVariableOpdense_7/BiasAdd/ReadVariableOp2>
dense_7/MatMul/ReadVariableOpdense_7/MatMul/ReadVariableOp2А
>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip>sequential_6/random_rotation_1/stateful_uniform/RngReadAndSkip:e a
1
_output_shapes
:€€€€€€€€€АА
,
_user_specified_namesequential_5_input
Ф
ь
C__inference_conv2d_22_layer_call_and_return_conditional_losses_7978

inputs8
conv2d_readvariableop_resource:  -
biasadd_readvariableop_resource: 
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€ 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€ 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€ 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:€€€€€€€€€ : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€ 
 
_user_specified_nameinputs"ћL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp* 
serving_defaultґ
[
sequential_5_inputE
$serving_default_sequential_5_input:0€€€€€€€€€АА;
dense_70
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict:юж
ЯЭ
layer-0
layer-1
layer_with_weights-0
layer-2
layer-3
layer_with_weights-1
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer-13
layer-14
layer_with_weights-6
layer-15
layer_with_weights-7
layer-16
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
В_default_save_signature
Г__call__
+Д&call_and_return_all_conditional_losses"ЬШ
_tf_keras_sequentialьЧ{"name": "sequential_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_5_input"}}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 256, "width": 256, "interpolation": "bilinear"}}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}]}}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 40, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 256, 256, 3]}, "float32", "sequential_5_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_7", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "sequential_5_input"}, "shared_object_id": 0}, {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 256, "width": 256, "interpolation": "bilinear"}}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}]}, "shared_object_id": 4}, {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}, "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 12}, {"class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 16}, {"class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 20}, {"class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 24}, {"class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 28}, {"class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 32}, {"class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 33}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": false}, "shared_object_id": 41}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 42}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
є
layer-0
layer-1
regularization_losses
	variables
trainable_variables
	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"О
_tf_keras_sequentialп{"name": "sequential_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 256, "width": 256, "interpolation": "bilinear"}}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}}]}, "shared_object_id": 4, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 256, 256, 3]}, "float32", "resizing_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_5", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "resizing_1_input"}, "shared_object_id": 1}, {"class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 256, "width": 256, "interpolation": "bilinear"}, "shared_object_id": 2}, {"class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "shared_object_id": 3}]}}}
и
layer-0
layer-1
 regularization_losses
!	variables
"trainable_variables
#	keras_api
З__call__
+И&call_and_return_all_conditional_losses"љ
_tf_keras_sequentialЮ{"name": "sequential_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}}]}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 256, 256, 3]}, "float32", "random_flip_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_6", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 256, 256, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "random_flip_1_input"}, "shared_object_id": 5}, {"class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}, "shared_object_id": 6}, {"class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}, "shared_object_id": 7}]}}}
б

$kernel
%bias
&regularization_losses
'	variables
(trainable_variables
)	keras_api
Й__call__
+К&call_and_return_all_conditional_losses"Ї

_tf_keras_layer†
{"name": "conv2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_18", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 44}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 256, 256, 3]}}
≥
*regularization_losses
+	variables
,trainable_variables
-	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_18", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_18", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 45}}
д

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
Н__call__
+О&call_and_return_all_conditional_losses"љ

_tf_keras_layer£
{"name": "conv2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_19", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 46}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 127, 127, 32]}}
≥
4regularization_losses
5	variables
6trainable_variables
7	keras_api
П__call__
+Р&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_19", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_19", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 47}}
в

8kernel
9bias
:regularization_losses
;	variables
<trainable_variables
=	keras_api
С__call__
+Т&call_and_return_all_conditional_losses"ї

_tf_keras_layer°
{"name": "conv2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_20", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 19, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 48}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 62, 62, 32]}}
≥
>regularization_losses
?	variables
@trainable_variables
A	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_20", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 49}}
в

Bkernel
Cbias
Dregularization_losses
E	variables
Ftrainable_variables
G	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"ї

_tf_keras_layer°
{"name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 23, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 50}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 30, 30, 32]}}
≥
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_21", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 51}}
в

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses"ї

_tf_keras_layer°
{"name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 52}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 14, 14, 32]}}
≥
Rregularization_losses
S	variables
Ttrainable_variables
U	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_22", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 53}}
а

Vkernel
Wbias
Xregularization_losses
Y	variables
Ztrainable_variables
[	keras_api
Э__call__
+Ю&call_and_return_all_conditional_losses"є

_tf_keras_layerЯ
{"name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 256, 256, 3]}, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 31, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 54}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 6, 6, 32]}}
≥
\regularization_losses
]	variables
^trainable_variables
_	keras_api
Я__call__
+†&call_and_return_all_conditional_losses"Ґ
_tf_keras_layerИ{"name": "max_pooling2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_23", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 55}}
Ш
`regularization_losses
a	variables
btrainable_variables
c	keras_api
°__call__
+Ґ&call_and_return_all_conditional_losses"З
_tf_keras_layerн{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 33, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 56}}
“

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
£__call__
+§&call_and_return_all_conditional_losses"Ђ
_tf_keras_layerС{"name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 34}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 35}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 57}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 128]}}
“

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
•__call__
+¶&call_and_return_all_conditional_losses"Ђ
_tf_keras_layerС{"name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 58}, "build_input_shape": {"class_name": "TensorShape", "items": [32, 64]}}
У
piter

qbeta_1

rbeta_2
	sdecay
tlearning_rate$mв%mг.mд/mе8mж9mзBmиCmйLmкMmлVmмWmнdmоemпjmрkmс$vт%vу.vф/vх8vц9vчBvшCvщLvъMvыVvьWvэdvюev€jvАkvБ"
	optimizer
 "
trackable_list_wrapper
Ц
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15"
trackable_list_wrapper
Ц
$0
%1
.2
/3
84
95
B6
C7
L8
M9
V10
W11
d12
e13
j14
k15"
trackable_list_wrapper
ќ
unon_trainable_variables
vlayer_metrics
regularization_losses
	variables
wmetrics
trainable_variables
xlayer_regularization_losses

ylayers
Г__call__
В_default_save_signature
+Д&call_and_return_all_conditional_losses
'Д"call_and_return_conditional_losses"
_generic_user_object
-
Іserving_default"
signature_map
М
z	keras_api"ъ
_tf_keras_layerа{"name": "resizing_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "Resizing", "config": {"name": "resizing_1", "trainable": true, "dtype": "float32", "height": 256, "width": 256, "interpolation": "bilinear"}, "shared_object_id": 2}
В
{	keras_api"р
_tf_keras_layer÷{"name": "rescaling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "Rescaling", "config": {"name": "rescaling_1", "trainable": true, "dtype": "float32", "scale": 0.00392156862745098, "offset": 0.0}, "shared_object_id": 3}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
±
|non_trainable_variables
}layer_metrics
regularization_losses
	variables
~metrics
trainable_variables
layer_regularization_losses
Аlayers
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
Њ
	Б_rng
В	keras_api"†
_tf_keras_layerЖ{"name": "random_flip_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "RandomFlip", "config": {"name": "random_flip_1", "trainable": true, "dtype": "float32", "mode": "horizontal_and_vertical", "seed": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 43}}
÷
	Г_rng
Д	keras_api"Є
_tf_keras_layerЮ{"name": "random_rotation_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": true, "class_name": "RandomRotation", "config": {"name": "random_rotation_1", "trainable": true, "dtype": "float32", "factor": 0.2, "fill_mode": "reflect", "fill_value": 0.0, "interpolation": "bilinear", "seed": null}, "shared_object_id": 7}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Еnon_trainable_variables
Жlayer_metrics
 regularization_losses
!	variables
Зmetrics
"trainable_variables
 Иlayer_regularization_losses
Йlayers
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_18/kernel
: 2conv2d_18/bias
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
µ
Кnon_trainable_variables
Лlayer_metrics
&regularization_losses
'	variables
Мmetrics
(trainable_variables
 Нlayer_regularization_losses
Оlayers
Й__call__
+К&call_and_return_all_conditional_losses
'К"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Пnon_trainable_variables
Рlayer_metrics
*regularization_losses
+	variables
Сmetrics
,trainable_variables
 Тlayer_regularization_losses
Уlayers
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_19/kernel
: 2conv2d_19/bias
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
µ
Фnon_trainable_variables
Хlayer_metrics
0regularization_losses
1	variables
Цmetrics
2trainable_variables
 Чlayer_regularization_losses
Шlayers
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Щnon_trainable_variables
Ъlayer_metrics
4regularization_losses
5	variables
Ыmetrics
6trainable_variables
 Ьlayer_regularization_losses
Эlayers
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_20/kernel
: 2conv2d_20/bias
 "
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
.
80
91"
trackable_list_wrapper
µ
Юnon_trainable_variables
Яlayer_metrics
:regularization_losses
;	variables
†metrics
<trainable_variables
 °layer_regularization_losses
Ґlayers
С__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
£non_trainable_variables
§layer_metrics
>regularization_losses
?	variables
•metrics
@trainable_variables
 ¶layer_regularization_losses
Іlayers
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_21/kernel
: 2conv2d_21/bias
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
µ
®non_trainable_variables
©layer_metrics
Dregularization_losses
E	variables
™metrics
Ftrainable_variables
 Ђlayer_regularization_losses
ђlayers
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
≠non_trainable_variables
Ѓlayer_metrics
Hregularization_losses
I	variables
ѓmetrics
Jtrainable_variables
 ∞layer_regularization_losses
±layers
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_22/kernel
: 2conv2d_22/bias
 "
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
µ
≤non_trainable_variables
≥layer_metrics
Nregularization_losses
O	variables
іmetrics
Ptrainable_variables
 µlayer_regularization_losses
ґlayers
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Јnon_trainable_variables
Єlayer_metrics
Rregularization_losses
S	variables
єmetrics
Ttrainable_variables
 Їlayer_regularization_losses
їlayers
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
*:(  2conv2d_23/kernel
: 2conv2d_23/bias
 "
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
.
V0
W1"
trackable_list_wrapper
µ
Љnon_trainable_variables
љlayer_metrics
Xregularization_losses
Y	variables
Њmetrics
Ztrainable_variables
 њlayer_regularization_losses
јlayers
Э__call__
+Ю&call_and_return_all_conditional_losses
'Ю"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
Ѕnon_trainable_variables
¬layer_metrics
\regularization_losses
]	variables
√metrics
^trainable_variables
 ƒlayer_regularization_losses
≈layers
Я__call__
+†&call_and_return_all_conditional_losses
'†"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
∆non_trainable_variables
«layer_metrics
`regularization_losses
a	variables
»metrics
btrainable_variables
 …layer_regularization_losses
 layers
°__call__
+Ґ&call_and_return_all_conditional_losses
'Ґ"call_and_return_conditional_losses"
_generic_user_object
!:	А@2dense_6/kernel
:@2dense_6/bias
 "
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
µ
Ћnon_trainable_variables
ћlayer_metrics
fregularization_losses
g	variables
Ќmetrics
htrainable_variables
 ќlayer_regularization_losses
ѕlayers
£__call__
+§&call_and_return_all_conditional_losses
'§"call_and_return_conditional_losses"
_generic_user_object
 :@2dense_7/kernel
:2dense_7/bias
 "
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
µ
–non_trainable_variables
—layer_metrics
lregularization_losses
m	variables
“metrics
ntrainable_variables
 ”layer_regularization_losses
‘layers
•__call__
+¶&call_and_return_all_conditional_losses
'¶"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
0
’0
÷1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ю
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16"
trackable_list_wrapper
"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
/
„
_state_var"
_generic_user_object
"
_generic_user_object
/
Ў
_state_var"
_generic_user_object
"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ў

ўtotal

Џcount
џ	variables
№	keras_api"Э
_tf_keras_metricВ{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 59}
£

Ёtotal

ёcount
я
_fn_kwargs
а	variables
б	keras_api"„
_tf_keras_metricЉ{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 42}
:	2Variable
:	2Variable
:  (2total
:  (2count
0
ў0
Џ1"
trackable_list_wrapper
.
џ	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
Ё0
ё1"
trackable_list_wrapper
.
а	variables"
_generic_user_object
/:- 2Adam/conv2d_18/kernel/m
!: 2Adam/conv2d_18/bias/m
/:-  2Adam/conv2d_19/kernel/m
!: 2Adam/conv2d_19/bias/m
/:-  2Adam/conv2d_20/kernel/m
!: 2Adam/conv2d_20/bias/m
/:-  2Adam/conv2d_21/kernel/m
!: 2Adam/conv2d_21/bias/m
/:-  2Adam/conv2d_22/kernel/m
!: 2Adam/conv2d_22/bias/m
/:-  2Adam/conv2d_23/kernel/m
!: 2Adam/conv2d_23/bias/m
&:$	А@2Adam/dense_6/kernel/m
:@2Adam/dense_6/bias/m
%:#@2Adam/dense_7/kernel/m
:2Adam/dense_7/bias/m
/:- 2Adam/conv2d_18/kernel/v
!: 2Adam/conv2d_18/bias/v
/:-  2Adam/conv2d_19/kernel/v
!: 2Adam/conv2d_19/bias/v
/:-  2Adam/conv2d_20/kernel/v
!: 2Adam/conv2d_20/bias/v
/:-  2Adam/conv2d_21/kernel/v
!: 2Adam/conv2d_21/bias/v
/:-  2Adam/conv2d_22/kernel/v
!: 2Adam/conv2d_22/bias/v
/:-  2Adam/conv2d_23/kernel/v
!: 2Adam/conv2d_23/bias/v
&:$	А@2Adam/dense_6/kernel/v
:@2Adam/dense_6/bias/v
%:#@2Adam/dense_7/kernel/v
:2Adam/dense_7/bias/v
т2п
__inference__wrapped_model_5554Ћ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *;Ґ8
6К3
sequential_5_input€€€€€€€€€АА
ъ2ч
+__inference_sequential_7_layer_call_fn_6703
+__inference_sequential_7_layer_call_fn_6740
+__inference_sequential_7_layer_call_fn_6779
+__inference_sequential_7_layer_call_fn_6818ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_7_layer_call_and_return_conditional_losses_6892
F__inference_sequential_7_layer_call_and_return_conditional_losses_7132
F__inference_sequential_7_layer_call_and_return_conditional_losses_7206
F__inference_sequential_7_layer_call_and_return_conditional_losses_7446ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ъ2ч
+__inference_sequential_5_layer_call_fn_7451
+__inference_sequential_5_layer_call_fn_7456
+__inference_sequential_5_layer_call_fn_7461
+__inference_sequential_5_layer_call_fn_7466ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_5_layer_call_and_return_conditional_losses_7476
F__inference_sequential_5_layer_call_and_return_conditional_losses_7486
F__inference_sequential_5_layer_call_and_return_conditional_losses_7496
F__inference_sequential_5_layer_call_and_return_conditional_losses_7506ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ъ2ч
+__inference_sequential_6_layer_call_fn_7511
+__inference_sequential_6_layer_call_fn_7516
+__inference_sequential_6_layer_call_fn_7523
+__inference_sequential_6_layer_call_fn_7530ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2г
F__inference_sequential_6_layer_call_and_return_conditional_losses_7534
F__inference_sequential_6_layer_call_and_return_conditional_losses_7704
F__inference_sequential_6_layer_call_and_return_conditional_losses_7708
F__inference_sequential_6_layer_call_and_return_conditional_losses_7878ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
“2ѕ
(__inference_conv2d_18_layer_call_fn_7887Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7898Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_18_layer_call_fn_6002а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_5996а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_conv2d_19_layer_call_fn_7907Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7918Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_19_layer_call_fn_6014а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_6008а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_conv2d_20_layer_call_fn_7927Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_20_layer_call_and_return_conditional_losses_7938Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_20_layer_call_fn_6026а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_6020а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_conv2d_21_layer_call_fn_7947Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_21_layer_call_and_return_conditional_losses_7958Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_21_layer_call_fn_6038а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_6032а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_conv2d_22_layer_call_fn_7967Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_22_layer_call_and_return_conditional_losses_7978Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_22_layer_call_fn_6050а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_6044а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_conv2d_23_layer_call_fn_7987Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_conv2d_23_layer_call_and_return_conditional_losses_7998Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
Ч2Ф
/__inference_max_pooling2d_23_layer_call_fn_6062а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
≤2ѓ
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6056а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
“2ѕ
(__inference_flatten_3_layer_call_fn_8003Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
н2к
C__inference_flatten_3_layer_call_and_return_conditional_losses_8009Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense_6_layer_call_fn_8018Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_6_layer_call_and_return_conditional_losses_8029Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
–2Ќ
&__inference_dense_7_layer_call_fn_8038Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
л2и
A__inference_dense_7_layer_call_and_return_conditional_losses_8049Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘B—
"__inference_signature_wrapper_6666sequential_5_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 ∞
__inference__wrapped_model_5554М$%./89BCLMVWdejkEҐB
;Ґ8
6К3
sequential_5_input€€€€€€€€€АА
™ "1™.
,
dense_7!К
dense_7€€€€€€€€€Ј
C__inference_conv2d_18_layer_call_and_return_conditional_losses_7898p$%9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ "/Ґ,
%К"
0€€€€€€€€€юю 
Ъ П
(__inference_conv2d_18_layer_call_fn_7887c$%9Ґ6
/Ґ,
*К'
inputs€€€€€€€€€АА
™ ""К€€€€€€€€€юю ≥
C__inference_conv2d_19_layer_call_and_return_conditional_losses_7918l./7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€}} 
Ъ Л
(__inference_conv2d_19_layer_call_fn_7907_./7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€}} ≥
C__inference_conv2d_20_layer_call_and_return_conditional_losses_7938l897Ґ4
-Ґ*
(К%
inputs€€€€€€€€€>> 
™ "-Ґ*
#К 
0€€€€€€€€€<< 
Ъ Л
(__inference_conv2d_20_layer_call_fn_7927_897Ґ4
-Ґ*
(К%
inputs€€€€€€€€€>> 
™ " К€€€€€€€€€<< ≥
C__inference_conv2d_21_layer_call_and_return_conditional_losses_7958lBC7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ Л
(__inference_conv2d_21_layer_call_fn_7947_BC7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€ ≥
C__inference_conv2d_22_layer_call_and_return_conditional_losses_7978lLM7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ Л
(__inference_conv2d_22_layer_call_fn_7967_LM7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€ ≥
C__inference_conv2d_23_layer_call_and_return_conditional_losses_7998lVW7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "-Ґ*
#К 
0€€€€€€€€€ 
Ъ Л
(__inference_conv2d_23_layer_call_fn_7987_VW7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ " К€€€€€€€€€ Ґ
A__inference_dense_6_layer_call_and_return_conditional_losses_8029]de0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "%Ґ"
К
0€€€€€€€€€@
Ъ z
&__inference_dense_6_layer_call_fn_8018Pde0Ґ-
&Ґ#
!К
inputs€€€€€€€€€А
™ "К€€€€€€€€€@°
A__inference_dense_7_layer_call_and_return_conditional_losses_8049\jk/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "%Ґ"
К
0€€€€€€€€€
Ъ y
&__inference_dense_7_layer_call_fn_8038Ojk/Ґ,
%Ґ"
 К
inputs€€€€€€€€€@
™ "К€€€€€€€€€®
C__inference_flatten_3_layer_call_and_return_conditional_losses_8009a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "&Ґ#
К
0€€€€€€€€€А
Ъ А
(__inference_flatten_3_layer_call_fn_8003T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€ 
™ "К€€€€€€€€€Ан
J__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_5996ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_18_layer_call_fn_6002СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_6008ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_19_layer_call_fn_6014СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_6020ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_20_layer_call_fn_6026СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_21_layer_call_and_return_conditional_losses_6032ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_21_layer_call_fn_6038СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_22_layer_call_and_return_conditional_losses_6044ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_22_layer_call_fn_6050СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€н
J__inference_max_pooling2d_23_layer_call_and_return_conditional_losses_6056ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ ≈
/__inference_max_pooling2d_23_layer_call_fn_6062СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Њ
F__inference_sequential_5_layer_call_and_return_conditional_losses_7476tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ Њ
F__inference_sequential_5_layer_call_and_return_conditional_losses_7486tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ »
F__inference_sequential_5_layer_call_and_return_conditional_losses_7496~KҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ »
F__inference_sequential_5_layer_call_and_return_conditional_losses_7506~KҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ †
+__inference_sequential_5_layer_call_fn_7451qKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААЦ
+__inference_sequential_5_layer_call_fn_7456gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААЦ
+__inference_sequential_5_layer_call_fn_7461gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€АА†
+__inference_sequential_5_layer_call_fn_7466qKҐH
AҐ>
4К1
resizing_1_input€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€ААЊ
F__inference_sequential_6_layer_call_and_return_conditional_losses_7534tAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ ¬
F__inference_sequential_6_layer_call_and_return_conditional_losses_7704xЎAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ ћ
F__inference_sequential_6_layer_call_and_return_conditional_losses_7708БNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p 

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ –
F__inference_sequential_6_layer_call_and_return_conditional_losses_7878ЕЎNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p

 
™ "/Ґ,
%К"
0€€€€€€€€€АА
Ъ £
+__inference_sequential_6_layer_call_fn_7511tNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААЦ
+__inference_sequential_6_layer_call_fn_7516gAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ ""К€€€€€€€€€ААЪ
+__inference_sequential_6_layer_call_fn_7523kЎAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€ААІ
+__inference_sequential_6_layer_call_fn_7530xЎNҐK
DҐA
7К4
random_flip_1_input€€€€€€€€€АА
p

 
™ ""К€€€€€€€€€АА∆
F__inference_sequential_7_layer_call_and_return_conditional_losses_6892|$%./89BCLMVWdejkAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ »
F__inference_sequential_7_layer_call_and_return_conditional_losses_7132~Ў$%./89BCLMVWdejkAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ”
F__inference_sequential_7_layer_call_and_return_conditional_losses_7206И$%./89BCLMVWdejkMҐJ
CҐ@
6К3
sequential_5_input€€€€€€€€€АА
p 

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ’
F__inference_sequential_7_layer_call_and_return_conditional_losses_7446КЎ$%./89BCLMVWdejkMҐJ
CҐ@
6К3
sequential_5_input€€€€€€€€€АА
p

 
™ "%Ґ"
К
0€€€€€€€€€
Ъ ™
+__inference_sequential_7_layer_call_fn_6703{$%./89BCLMVWdejkMҐJ
CҐ@
6К3
sequential_5_input€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€Ю
+__inference_sequential_7_layer_call_fn_6740o$%./89BCLMVWdejkAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p 

 
™ "К€€€€€€€€€†
+__inference_sequential_7_layer_call_fn_6779qЎ$%./89BCLMVWdejkAҐ>
7Ґ4
*К'
inputs€€€€€€€€€АА
p

 
™ "К€€€€€€€€€ђ
+__inference_sequential_7_layer_call_fn_6818}Ў$%./89BCLMVWdejkMҐJ
CҐ@
6К3
sequential_5_input€€€€€€€€€АА
p

 
™ "К€€€€€€€€€…
"__inference_signature_wrapper_6666Ґ$%./89BCLMVWdejk[ҐX
Ґ 
Q™N
L
sequential_5_input6К3
sequential_5_input€€€€€€€€€АА"1™.
,
dense_7!К
dense_7€€€€€€€€€