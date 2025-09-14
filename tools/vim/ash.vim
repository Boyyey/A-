" A# Language Vim Integration
" Provides syntax highlighting, indentation, and language features

if exists("b:did_ash_ftplugin")
    finish
endif
let b:did_ash_ftplugin = 1

" Set file type
set filetype=ash

" Syntax highlighting
syntax keyword ashKeyword fn let mut if else match loop break continue return
syntax keyword ashKeyword type enum struct trait impl mod use pub
syntax keyword ashKeyword actor message spawn channel async await
syntax keyword ashKeyword move borrow ref mut static
syntax keyword ashKeyword IO Concurrency Resource Pure

" Ownership and borrowing
syntax keyword ashOwnership ! & mut move borrow
syntax match ashLifetime "'[a-zA-Z_][a-zA-Z0-9_]*"
syntax match ashLifetime "'static"

" Types
syntax keyword ashType i32 i64 f32 f64 bool String Vec Option Result
syntax keyword ashType Tensor NeuralNetwork Model Dataset Optimizer

" ML/AI keywords
syntax keyword ashML create_tensor create_model train_model predict
syntax keyword ashML forward backward optimizer_step compute_loss
syntax keyword ashML TENSOR_FLOAT32 TENSOR_FLOAT64 TENSOR_INT32
syntax keyword ashML LAYER_DENSE LAYER_CONV2D LAYER_LSTM LAYER_ATTENTION
syntax keyword ashML OPTIMIZER_SGD OPTIMIZER_ADAM OPTIMIZER_RMSPROP
syntax keyword ashML LOSS_MSE LOSS_CROSSENTROPY LOSS_BINARY_CROSSENTROPY
syntax keyword ashML ACTIVATION_RELU ACTIVATION_SIGMOID ACTIVATION_TANH

" Comments
syntax match ashComment "//.*$"
syntax region ashComment start="/\*" end="\*/"

" Strings
syntax region ashString start='"' end='"' skip='\\"'
syntax region ashString start='r"' end='"'

" Numbers
syntax match ashNumber "\<[0-9]\+\([eE][+-]\?[0-9]\+\)\?\(f32\|f64\)\?\>"
syntax match ashNumber "\<0x[0-9a-fA-F]\+\>"
syntax match ashNumber "\<0o[0-7]\+\>"
syntax match ashNumber "\<0b[01]\+\>"

" Operators
syntax match ashOperator "+\|-\|*\|\/\|%\|\|\|\|\&\&\|!\|\|\|\|\&\|\|\|\|\^\|\~\|\<\|\>\|\<=\|\>=\|\==\|\!="
syntax match ashOperator "=\|+\|-\|*\|\/\|%"

" Punctuation
syntax match ashPunctuation "[\[\]{}();,.]"

" Functions
syntax match ashFunction "\<[a-zA-Z_][a-zA-Z0-9_]*\s*("

" Highlighting groups
highlight link ashKeyword Keyword
highlight link ashOwnership Special
highlight link ashLifetime Identifier
highlight link ashType Type
highlight link ashML Function
highlight link ashComment Comment
highlight link ashString String
highlight link ashNumber Number
highlight link ashOperator Operator
highlight link ashPunctuation Delimiter
highlight link ashFunction Function

" Indentation
setlocal indentexpr=ash#Indent()
setlocal indentkeys=0{,0},0),0],0),0#,!^F,o,O,e

" Compiler integration
command! -buffer AshCompile call ash#Compile()
command! -buffer AshRun call ash#Run()
command! -buffer AshCheck call ash#Check()
command! -buffer AshFormat call ash#Format()

" Auto-completion
setlocal omnifunc=ash#Complete

" Folding
setlocal foldmethod=syntax
setlocal foldlevelstart=1

" Error highlighting
let g:ash_error_highlight = 1
 
if exists("b:current_syntax") 
  finish 
endif 
 
syn keyword ashKeyword fn mod struct enum trait impl actor message 
syn keyword ashKeyword let mut if else match for while loop break continue 
syn keyword ashKeyword return use pub priv static const 
syn keyword ashKeyword true false null 
 
syn keyword ashType i32 i64 f32 f64 bool String Vec Option Result 
syn keyword ashType Tensor Model NeuralNetwork Optimizer 
 
syn keyword ashEffect Pure IO Resource Concurrency 
 
syn match ashComment "//.*$" 
syn region ashComment start="/\*" end="\*/" 
 
syn region ashString start=+"+ end=+"+ 
 
syn match ashNumber "\<[0-9]\+\>" 
syn match ashNumber "\<[0-9]\+\.[0-9]\+" 
 
syn match ashOperator "[+\-*/=<>!&|^]" 
syn match ashOperator "->" 
syn match ashOperator "@" 
 
hi def link ashKeyword Keyword 
hi def link ashType Type 
hi def link ashEffect Special 
hi def link ashComment Comment 
hi def link ashString String 
hi def link ashNumber Number 
hi def link ashOperator Operator 
 
let b:current_syntax = "ash" 
