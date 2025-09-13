(* A# Core Calculus - Formal Semantics in Coq *)
(* Research Language with Formal Verification *)

Require Import Coq.Arith.Arith.
Require Import Coq.Lists.List.
Require Import Coq.Strings.String.
Require Import Coq.Logic.FunctionalExtensionality.

(* Syntax Definition *)

Inductive TypeVar : Type :=
  | TVar : string -> TypeVar.

Inductive Lifetime : Type :=
  | LVar : string -> Lifetime
  | LStatic : Lifetime.

Inductive Type_ : Type :=
  | TInt : Type_
  | TBool : Type_
  | TString : Type_
  | TUnit : Type_
  | TRef : Lifetime -> Type_ -> Type_
  | TMutRef : Lifetime -> Type_ -> Type_
  | TOwned : Type_ -> Type_
  | TArray : Type_ -> nat -> Type_
  | TTuple : list Type_ -> Type_
  | TFun : list Type_ -> Type_ -> Effect -> Type_
  | TForall : list TypeVar -> Type_ -> Type_
  | TExists : list TypeVar -> Type_ -> Type_
  | TVar : TypeVar -> Type_
  | TApp : Type_ -> list Type_ -> Type_

with Effect : Type :=
  | EPure : Effect
  | EIO : Effect
  | EConcurrency : Effect
  | EResource : Effect
  | EUnion : Effect -> Effect -> Effect.

Inductive Expr : Type :=
  | EInt : nat -> Expr
  | EBool : bool -> Expr
  | EString : string -> Expr
  | EUnit : Expr
  | EVar : string -> Expr
  | EApp : Expr -> list Expr -> Expr
  | ELam : list (string * Type_) -> Expr -> Expr
  | ELet : string -> Type_ -> Expr -> Expr -> Expr
  | EIf : Expr -> Expr -> Expr -> Expr
  | EMatch : Expr -> list (Pattern * Expr) -> Expr
  | ERef : Lifetime -> Expr -> Expr
  | EMutRef : Lifetime -> Expr -> Expr
  | EDeref : Expr -> Expr
  | EAssign : Expr -> Expr -> Expr
  | EMove : Expr -> Expr
  | EBorrow : Expr -> Expr
  | EField : Expr -> string -> Expr
  | EIndex : Expr -> Expr -> Expr
  | EArray : list Expr -> Expr
  | ETuple : list Expr -> Expr
  | ETypeApp : Expr -> list Type_ -> Expr
  | ETypeLam : list TypeVar -> Expr -> Expr

with Pattern : Type :=
  | PVar : string -> Pattern
  | PWildcard : Pattern
  | PLit : Expr -> Pattern
  | PTuple : list Pattern -> Pattern
  | PRef : Pattern -> Pattern
  | PMutRef : Pattern -> Pattern
  | POwned : Pattern -> Pattern.

(* Type Environment *)

Definition TypeEnv := list (string * Type_).

Definition LifetimeEnv := list (string * Lifetime).

Definition EffectEnv := list (string * Effect).

(* Typing Rules *)

Reserved Notation "Gamma ';' Delta ';' Phi '|-' e ':' t ';' e'" (at level 50).

Inductive Typing : TypeEnv -> LifetimeEnv -> EffectEnv -> Expr -> Type_ -> Effect -> Prop :=
  | TInt : forall Gamma Delta Phi n,
      Gamma; Delta; Phi |- EInt n : TInt; EPure
  | TBool : forall Gamma Delta Phi b,
      Gamma; Delta; Phi |- EBool b : TBool; EPure
  | TString : forall Gamma Delta Phi s,
      Gamma; Delta; Phi |- EString s : TString; EPure
  | TUnit : forall Gamma Delta Phi,
      Gamma; Delta; Phi |- EUnit : TUnit; EPure
  | TVar : forall Gamma Delta Phi x t,
      In (x, t) Gamma ->
      Gamma; Delta; Phi |- EVar x : t; EPure
  | TApp : forall Gamma Delta Phi e es t ts e' es',
      Gamma; Delta; Phi |- e : TFun ts t e'; EPure ->
      Forall2 (fun e t => Gamma; Delta; Phi |- e : t; EPure) es ts ->
      Gamma; Delta; Phi |- EApp e es : t; e'
  | TLam : forall Gamma Delta Phi xs e t ts e',
      (map fst xs ++ Gamma); Delta; Phi |- e : t; e' ->
      Gamma; Delta; Phi |- ELam xs e : TFun (map snd xs) t e'; EPure
  | TLet : forall Gamma Delta Phi x t e1 e2 t' e1' e2',
      Gamma; Delta; Phi |- e1 : t; e1' ->
      ((x, t) :: Gamma); Delta; Phi |- e2 : t'; e2' ->
      Gamma; Delta; Phi |- ELet x t e1 e2 : t'; EUnion e1' e2'
  | TIf : forall Gamma Delta Phi e1 e2 e3 t e',
      Gamma; Delta; Phi |- e1 : TBool; EPure ->
      Gamma; Delta; Phi |- e2 : t; e' ->
      Gamma; Delta; Phi |- e3 : t; e' ->
      Gamma; Delta; Phi |- EIf e1 e2 e3 : t; e'
  | TRef : forall Gamma Delta Phi e t l,
      Gamma; Delta; Phi |- e : t; EPure ->
      In (l, l) Delta ->
      Gamma; Delta; Phi |- ERef l e : TRef l t; EPure
  | TMutRef : forall Gamma Delta Phi e t l,
      Gamma; Delta; Phi |- e : t; EPure ->
      In (l, l) Delta ->
      Gamma; Delta; Phi |- EMutRef l e : TMutRef l t; EPure
  | TDeref : forall Gamma Delta Phi e t l,
      Gamma; Delta; Phi |- e : TRef l t; EPure ->
      Gamma; Delta; Phi |- EDeref e : t; EPure
  | TAssign : forall Gamma Delta Phi e1 e2 t l,
      Gamma; Delta; Phi |- e1 : TMutRef l t; EPure ->
      Gamma; Delta; Phi |- e2 : t; EPure ->
      Gamma; Delta; Phi |- EAssign e1 e2 : TUnit; EPure
  | TMove : forall Gamma Delta Phi e t,
      Gamma; Delta; Phi |- e : TOwned t; EPure ->
      Gamma; Delta; Phi |- EMove e : t; EPure
  | TBorrow : forall Gamma Delta Phi e t,
      Gamma; Delta; Phi |- e : t; EPure ->
      Gamma; Delta; Phi |- EBorrow e : TRef (LVar "r") t; EPure
  | TField : forall Gamma Delta Phi e t f,
      Gamma; Delta; Phi |- e : t; EPure ->
      Gamma; Delta; Phi |- EField e f : t; EPure
  | TIndex : forall Gamma Delta Phi e1 e2 t,
      Gamma; Delta; Phi |- e1 : TArray t 0; EPure ->
      Gamma; Delta; Phi |- e2 : TInt; EPure ->
      Gamma; Delta; Phi |- EIndex e1 e2 : t; EPure
  | TArray : forall Gamma Delta Phi es t,
      Forall (fun e => Gamma; Delta; Phi |- e : t; EPure) es ->
      Gamma; Delta; Phi |- EArray es : TArray t (length es); EPure
  | TTuple : forall Gamma Delta Phi es ts,
      Forall2 (fun e t => Gamma; Delta; Phi |- e : t; EPure) es ts ->
      Gamma; Delta; Phi |- ETuple es : TTuple ts; EPure
  | TTypeApp : forall Gamma Delta Phi e ts t t' e',
      Gamma; Delta; Phi |- e : TForall (map TVar (map fst ts)) t; e' ->
      Gamma; Delta; Phi |- ETypeApp e (map snd ts) : t'; e'
  | TTypeLam : forall Gamma Delta Phi xs e t e',
      Gamma; Delta; Phi |- e : t; e' ->
      Gamma; Delta; Phi |- ETypeLam xs e : TForall xs t; e'

where "Gamma ';' Delta ';' Phi '|-' e ':' t ';' e'" := (Typing Gamma Delta Phi e t e).

(* Operational Semantics *)

Inductive Value : Expr -> Prop :=
  | VInt : forall n, Value (EInt n)
  | VBool : forall b, Value (EBool b)
  | VString : forall s, Value (EString s)
  | VUnit : Value EUnit
  | VRef : forall l e, Value e -> Value (ERef l e)
  | VMutRef : forall l e, Value e -> Value (EMutRef l e)
  | VLam : forall xs e, Value (ELam xs e)
  | VArray : forall es, Forall Value es -> Value (EArray es)
  | VTuple : forall es, Forall Value es -> Value (ETuple es).

Reserved Notation "e '-->' e'" (at level 50).

Inductive Step : Expr -> Expr -> Prop :=
  | SApp : forall xs e es vs,
      Forall Value vs ->
      length vs = length xs ->
      EApp (ELam xs e) es --> subst_expr (zip xs vs) e
  | SLet : forall x t e1 e2 v,
      Value v ->
      ELet x t e1 e2 --> subst_expr [(x, v)] e2
  | SIfTrue : forall e1 e2 e3,
      EIf (EBool true) e1 e2 --> e1
  | SIfFalse : forall e1 e2 e3,
      EIf (EBool false) e1 e2 --> e2
  | SDeref : forall l e v,
      Value v ->
      EDeref (ERef l v) --> v
  | SAssign : forall l e v,
      Value v ->
      EAssign (EMutRef l e) v --> EUnit
  | SMove : forall e v,
      Value v ->
      EMove (TOwned v) --> v
  | SBorrow : forall e v,
      Value v ->
      EBorrow v --> ERef (LVar "r") v
  | SField : forall e f v,
      Value v ->
      EField e f --> v
  | SIndex : forall es i v,
      Value v ->
      nth i es v = v ->
      EIndex (EArray es) (EInt i) --> v
  | STypeApp : forall xs e ts,
      ETypeApp (ETypeLam xs e) ts --> subst_type_expr (zip xs ts) e

where "e '-->' e'" := (Step e e').

(* Progress and Preservation Theorems *)

Theorem Progress : forall Gamma Delta Phi e t e',
  Gamma; Delta; Phi |- e : t; e' ->
  Value e \/ exists e'', e --> e''.
Proof.
  intros Gamma Delta Phi e t e' H.
  induction H; try (left; constructor; assumption).
  - right. destruct IH as [Hv | [e'' Hstep]].
    + destruct Hv as [n | b | s | | | | | | |].
      * exists (EInt n). constructor.
      * exists (EBool b). constructor.
      * exists (EString s). constructor.
      * exists EUnit. constructor.
      * exists (EVar x). constructor.
      * exists (EApp e es). constructor.
      * exists (ELam xs e). constructor.
      * exists (EArray es). constructor.
      * exists (ETuple es). constructor.
    + exists e''. assumption.
  - right. destruct IH as [Hv | [e'' Hstep]].
    + destruct Hv as [n | b | s | | | | | | |].
      * exists (EInt n). constructor.
      * exists (EBool b). constructor.
      * exists (EString s). constructor.
      * exists EUnit. constructor.
      * exists (EVar x). constructor.
      * exists (EApp e es). constructor.
      * exists (ELam xs e). constructor.
      * exists (EArray es). constructor.
      * exists (ETuple es). constructor.
    + exists e''. assumption.
Qed.

Theorem Preservation : forall Gamma Delta Phi e t e' e'',
  Gamma; Delta; Phi |- e : t; e' ->
  e --> e'' ->
  Gamma; Delta; Phi |- e'' : t; e'.
Proof.
  intros Gamma Delta Phi e t e' e'' H Hstep.
  induction H; inversion Hstep; subst; try constructor; assumption.
Qed.

(* Memory Safety Properties *)

Definition MemorySafe (e : Expr) : Prop :=
  forall e', e -->* e' -> ~ (exists e'', e' --> e'').

Theorem TypeSafety : forall Gamma Delta Phi e t e',
  Gamma; Delta; Phi |- e : t; e' ->
  MemorySafe e.
Proof.
  intros Gamma Delta Phi e t e' H.
  unfold MemorySafe.
  intros e' Hstep.
  destruct Hstep as [e' Hstep].
  apply Progress in H.
  destruct H as [Hv | [e'' Hstep']].
  - inversion Hv.
  - exists e''. assumption.
Qed.

(* Ownership and Borrowing Properties *)

Definition Owned (e : Expr) : Prop :=
  exists t, e = TOwned t.

Definition Borrowed (e : Expr) : Prop :=
  exists l t, e = TRef l t \/ e = TMutRef l t.

Theorem OwnershipInvariant : forall Gamma Delta Phi e t e',
  Gamma; Delta; Phi |- e : t; e' ->
  Owned e \/ Borrowed e \/ Value e.
Proof.
  intros Gamma Delta Phi e t e' H.
  induction H; try (right; right; constructor; assumption).
  - left. exists t. reflexivity.
  - right. left. exists l, t. left. reflexivity.
  - right. left. exists l, t. right. reflexivity.
  - right. right. constructor. assumption.
Qed.

(* Resource Guarantees *)

Definition ResourceBound (e : Expr) (n : nat) : Prop :=
  forall e', e -->* e' -> length (free_vars e') <= n.

Theorem ResourceSafety : forall Gamma Delta Phi e t e',
  Gamma; Delta; Phi |- e : t; e' ->
  exists n, ResourceBound e n.
Proof.
  intros Gamma Delta Phi e t e' H.
  exists (length (free_vars e)).
  intros e' Hstep.
  apply Preservation in H.
  assumption.
Qed.
