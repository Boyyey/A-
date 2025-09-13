# A# Language Specification

## Core Calculus

### Syntax

```
Program ::= Module*

Module ::= mod Name { Item* }

Item ::= TypeDecl | FnDecl | ActorDecl | TraitDecl

TypeDecl ::= type Name<TypeParams> = TypeExpr
           | enum Name<TypeParams> { Variant* }

Variant ::= Name(TypeExpr*)

TypeExpr ::= TypeName<TypeArgs>
           | TypeExpr -> TypeExpr
           | &'r TypeExpr
           | &'r mut TypeExpr
           | !TypeExpr
           | [TypeExpr; N]
           | (TypeExpr*)

TypeParams ::= <Name*>
TypeArgs ::= <TypeExpr*>

FnDecl ::= fn Name<TypeParams>('r*)(Params) -> TypeExpr @ Effects { Block }

Params ::= (Param*)
Param ::= Name: TypeExpr

Block ::= { Stmt* Expr? }

Stmt ::= let Pattern = Expr;
        | Expr;

Expr ::= Literal
       | Name
       | Expr(Args)
       | Expr.field
       | Expr[Index]
       | Expr @ Region
       | &'r Expr
       | &'r mut Expr
       | *Expr
       | Expr.borrow()
       | Expr.move()
       | if Expr { Block } else { Block }
       | match Expr { Pattern => Expr* }
       | loop { Block }
       | break Expr?
       | continue
       | return Expr?

Pattern ::= Name
          | Literal
          | Pattern(Pattern*)
          | Pattern @ Region
          | _ (wildcard)

Literal ::= Integer | Float | String | Boolean

Effects ::= IO | Concurrency | Resource | Pure

Region ::= 'r | 'static | 'local
```

### Type System

#### Ownership Types
- `!T` - Unique ownership (moved on assignment)
- `&'r T` - Immutable borrow with region `'r`
- `&'r mut T` - Mutable borrow with region `'r`
- `T` - Copyable type (no ownership)

#### Region Polymorphism
- Functions can be parameterized by regions
- Regions track lifetime relationships
- Prevents use-after-free and data races

#### Effect Types
- `@ IO` - May perform I/O operations
- `@ Concurrency` - May spawn tasks or use channels
- `@ Resource` - May allocate/deallocate memory
- `@ Pure` - No side effects (default)

### Operational Semantics

#### Memory Model
- Stack-allocated values with move semantics
- Region-based heap allocation
- No garbage collection - explicit resource management
- Ownership transfer prevents double-free

#### Concurrency Model
- Actor-based message passing
- Linear capabilities for channel communication
- Session types for protocol verification
- No shared mutable state

### Formal Properties

#### Type Safety
- Progress: Well-typed expressions don't get stuck
- Preservation: Evaluation preserves types
- Memory safety: No use-after-free or double-free
- Data race freedom: No concurrent access to mutable data

#### Resource Guarantees
- Stack usage bounded by type annotations
- Heap allocation tracked by region types
- I/O operations tracked by effect types
- Deterministic resource consumption

## Implementation Strategy

### Phase 1: Core Language
- Lexer and parser for basic syntax
- Type checker with ownership inference
- Simple IR and LLVM code generation

### Phase 2: Advanced Features
- Region polymorphism
- Concurrency primitives
- Effect system

### Phase 3: Verification
- Formal semantics in Coq
- Mechanized proofs for type safety
- Verified compiler passes

### Phase 4: Optimization
- Ownership-aware optimizations
- Region-based memory management
- ML-guided code generation
