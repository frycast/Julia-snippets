#= Resources
* Mike Innes stuff is great, including 'The Many Types of Types'
  https://mikeinnes.github.io/2020/05/19/types.html

* DataFrames indexing https://dataframes.juliadata.org/stable/lib/indexing/#getindex-and-view-1
=#

## %%
# Use || rather than | for short circuit evaluation
# true   | error()  # returns error
# false  & error()  # returns error
true  || error()  # doesn't return error
false && error()  # doesn't return error

## %%
# using short circuit evaluation for compact if statements
x = 2; (x > 0) && (x = 1) # x = 1

## %%
# Make a dictionary and check if it has a key
d = Dict{String, Float64}()
haskey(d, "mykey")

## %% 
# map with side effect over dictionary values
d = Dict("a" => 1.0, "b" => 2.0)
map!(v -> v/5, values(d))

## %%
# Enumerate function works on arrays and dictionaries
for (i,x) in enumerate(['a','b','c']) 
    println((i,x))
end
for (k,v) in enumerate(Dict("a" => 1, "b" => 2)) 
  println((k,v))
end


## %%
# Index of element in array 
findall(x -> x == 0, [0,1,2,0])

## %%
# map and filter
map(x -> x == 0, [0,1,2,0])
filter(x -> x == 0, [0,1,2,0])

## %%
# Dump contents (like str in R)
expr2 = Dict("a" => 1.0, "b" => 2.0)
dump(expr2)
struct MyStruct
  foo::Int
  bar::String
end
m = MyStruct(1, "a")
dump(m)

## %%
# Find out which function is called
@which 1/1
@edit 1.0/1.0

## %%
# Find out how many methods are defined
methods(*)

## %%
# Lowering code all the way to machine code
@code_lowered ==(true, true)
@code_typed ==(true, true)
@code_llvm ==(true, true)
@code_native ==(true, true)

## %%
# Fix the second argument to x with Fix2 (similar to y->f(y, x))
function add(x,y)
    x + y
end
add(x) = Base.Fix2(add, x)
add_one = add(1)

## %%
# apply_funcs1 does the same thing as apply_funcs2
# apply an array of functions
using Statistics
function apply_funcs1(funcs; x = 1:100)
  y = zeros(length(funcs))
  for i in 1:length(funcs)
    y[i] = funcs[i](x)
  end
  return y
end
apply_funcs2(funcs; x = 1:100) = map(func -> func(x), funcs)
apply_funcs1([mean, median, cor, sum])
apply_funcs2([mean, median, cor, sum])

## %%
# Using multinomial coefficients.
# Task: given 20 people, what is the probability that 
#       4 months have 0 birthdays, 4 have 2 birthdays and 4 have 3 birthdays.
# Solution:
using Combinatorics

# %1 = number of ways to allocate 20 people into specific month configuration (2,2,2,2,3,3,3,3,0,0,0,0)
one = multinomial(2,2,2,2,3,3,3,3)
# %2 = %1 * ( probability of any specific configuration )
two = one / 12.0^20
# solution = %2 * ( number of acceptable specific month configurations )
solution = two * multinomial(4,4,4)
# Notice also some messier ways to calculate the above coefficients:
factorial(20) / (2^4*6^4) == multinomial(2,2,2,2,3,3,3,3)
binomial(12,4) * binomial(8,4) == multinomial(4,4,4)
multinomial(2,2,2,2,3,3,3,3) == 
  binomial(20, 2) * binomial(18, 2) * 
  binomial(16, 2) * binomial(14, 2) * 
  binomial(12, 3) * binomial(9, 3) * 
  binomial(6, 3) * binomial(3, 3)

## %%
# Any new constructor will override the default constructor, but the default can be recreated
mutable struct Node
  neighbours::Vector{Node}
  ID::Int64
  Node(neighbours::Vector{Node}, ID::Int64) = new(neighbours, ID)
  Node(ID::Int64) = new(Vector{Node}(), ID)
  Node(ID::Int64, ID2::Int64) = new(Vector{Node}(), ID + ID2)
end

## %%
# Using a tree graph from the node structure defined in the 
# previous snippet, we can define a depth first search (DFS) 
# and breadth first search (BFS) using only the difference 
# between stack and queue. Note that the stack can also 
# be easily implemented with recursion.
using DataStructures
graph = [Node(i) for i in 1:7]
graph[1].neighbours = [graph[2], graph[5]]
graph[2].neighbours = [graph[3], graph[4]]
graph[5].neighbours = [graph[6], graph[7]]
function XFS(n::Node, data_str, fun_in, fun_out) # This one is user defined!
  s = data_str{Node}()
  fun_in(s, n)
  while(length(s) > 0)
    println(top(s).ID)
    temp = fun_out(s)
    for neighbour in temp.neighbours
      fun_in(s, neighbour)
    end
  end
end
XFS(graph[1], Stack, push!, pop!)        # DFS
XFS(graph[1], Queue, enqueue!, dequeue!) # BFS

## %%
# Enums are a thing
@enum FRUIT apple=3 orange=1 kiwi=2
@enum Colour red green blue
Int(apple)
instances(Colour)
Int(red)

## %%
# Sorting algorithms
b = [9,10,8,2,4,11]
function swap!(arr, i, j)
  temp = arr[i]; arr[i] = arr[j]; arr[j] = temp
  return nothing
end
# Bubble sort
b = [9,10,8,2,4,11]
for i in 1:length(b)-1, j in 1:length(b)-i
  if b[j] > b[j+1] swap!(b, j, j+1) end
end
b
# Selection sort
b = [9,10,8,2,4,11]
for j in 1:length(b)-1
  mi = length(b)
  for i in j:length(b)-1
    if b[i] < b[mi] mi = i end
  end
  swap!(b,j,mi)
end
b
# Insertion sort (can be done much better)
b = [9,10,8,1,2,4,7,3]
for i in 2:length(b)
  bi = b[i]; j = i
  for j in i-1:-1:0
    if (j == 0 || b[j] < bi) b[j+1] = bi; break end
    b[j+1] = b[j]
  end
end 
println(b)

## %%
# Access last elements of vector
v = [1,2,3]
v[end]
v[end-1]

## %%
# Parametric types
struct Point{T}
  x::T
  y::T
end
mypoint = Point(1,2)
isa(mypoint.x, Int64)
mypoint = Point("a","b")
# mypoint = Point(1,"b") # error
struct Point2
  x
  y
end
mypoint = Point2(1,1)
isa(mypoint, Any)
mypoint = Point2(1,"string")
struct Point3{T,W}
  x::T
  y::W
end
mypoint = Point3(1,1)
mypoint = Point3(1,"string")

## %%
# Extend function on a parametric type
import Base: angle
angle(p::Point) = atan(p.y, p.x)
# angle(Point("b","a")) # error

## %%
# @code_typed can highlight when unnecessary work is being done
struct Point{T} x::T; y::T end
import Base.angle
angle(p::Point) = atan(p.y, p.x)
@code_typed angle(Point(1,2)) # extra conversion from int to float
@code_typed angle(Point(2.0,1.0))

## %% 
# Defining a union of types can preserve inheritance
IntFloat = Union{Int64, Float64}
IntFloat <: Number
NumString = Union{Int64, Float64, String}
NumString <: Number

## %%
# DIY abstract type
abstract type AbstractPoint end
struct MyPoint <: AbstractPoint x; y end
MyTheta(p::AbstractPoint) = p.x + p.y

## %%
# Abstract type example
abstract type Clothing end
struct Shoe <: Clothing
  laces::Bool
  size::Int8
end
struct Shirt <: Clothing
  image::String
  size::Int8
end
SizeMatch(c::Clothing, s::Int) = (c.size == s)

## %% Broadcasting the . operator (selection operator) is done with getfield
myshoes = [Shoe(true, i) for i in 1:10]
@code_lowered myshoes[1].size
getfield.(myshoes, :size) # remember to use Symbol type here

## %% Broadcasting the . operator can be done more easily with nested .
get_size(myshoe) = myshoe.size
get_size.(myshoes)

## %% function vs begin block vs let block
function f()
  # Always returns one object.
  # Variables have local scope
  nothing
end
a = begin
  # Grouped expression and return value of last subexpression
  # Variable are not local
  nothing 
end
b = let
  # Grouped just like begin
  # Variables have local scope
  nothing
end

## %% Using a do block to write the first agument of a function like map
map(x -> x+1, [1,2,3])
map([1,2,3]) do x
  x + 1
end

## %% Another example of a do block
function f(func, y)
 func(y)
end
f("hi") do x
  x * " my name is Danny"
end

## %% Map is also a quick way to return an array of arrays
map(1:3) do x
  [1 2; 3 4] .+ x
end
map(1:3) do _
  [1 2; 3 4]
end

## % explanation of @view, @views and slices
# Indexing (with range or integer) creates a copy (slice)
a = [1,2,3]
b = a[1:2]
b[1] = 2
a
# Array assignment creates a reference 
a = [1,2,3]
b = a
b[1] = 2
a
# view can create a reference to a range (a view)
a = [1,2,3]
b = view(a, 1:2)
b[1] = 2
a
# @view macro can be used instead of view
a = [1,2,3]
b = @view a[1:2]
b[1] = 2
a
# @views macro converts block of code swapping slices for views
a = [1,2,3]
@views begin
  b = a[1:2]
  c = a[1:2]
end
@views d = a[1:2]
b[1] = 2
a
c[1] = 3
a
d[1] = 4
a

## %% @. macro converts blocks of code to broadcast
# but be wary it converts every function call, even =
# hence arr1 and arr2 need to be defined in advance here:
arr1, arr2 = zeros(3), zeros(3)
@. begin
  arr1 = [1,2,3] ^ 2
  arr2 = [1,2,3] ^ 3
end
arr1, arr2

## %% view macro for removing unnecessary allocations
# and also using the @. macro to vectorise.
function heavy(n, vec1, vec2)
  for i in 1:2:n-1
    vec1[i:i+1] = vec2[i:i+1] .^ i
  end
  return nothing
end
function light(
  n::Int64, vec1::Vector{Float64}, vec2::Vector{Float64})
  for i in 1:2:n-1
    @views @. vec1[i:i+1] = vec2[i:i+1] ^ i
  end
  return nothing
end
using BenchmarkTools
n = 10_000
vec1, vec2 = zeros(n), randn(n)
@benchmark heavy(n, vec1, vec2)
@benchmark light(n, vec1, vec2)


## %% An array of named tuples can have
# its tuples summed elementwise using sum
# with getfield
A = [(S = [1,2],I = [1,2]),
     (S = [3,4],I = [3,4]),
     (S = [5,6],I = [5,6])]
sum(x->x.I, A) / length(A)


## %% append to end of array
a = [1,2,3]
append!(a, [1,2,3])

## %% There are many infixable Julia function names
# See the list: https://github.com/JuliaLang/julia/blob/master/src/julia-parser.scm#L23-L24
⊕(a,b) = a .+ b
[1,2,3] ⊕ [1,2,3]

## %% rand can be used to randomly return an element of a vector
rand([1,2,3])

## %% accumulate works like R cumsum, but is more general, using any accumulator
accumulate(+, 1:10)

## %% a bounding box [-L,L]x[-L,L] boundary collision can
# be created by exploiting 2 kinds of symmetry:
struct Coordinate
  x::Int64
  y::Int64
end
Coordinate() = Coordinate(0,0)
function collide_boundary(x::Int64, L::Number)
  if (abs(x) > L) x = sign(x)*L end
  return x	
end
function collide_boundary(c::Coordinate, L::Number)
  return Coordinate(collide_boundary(c.x, L), collide_boundary(c.y, L))
end

## %% To overload the + function in Base we need to use the Symbol :+
function Base.:+(a::Coordinate, b::Coordinate)
  return Coordinate(a.x + b.x, a.y + b.y)
end

## %% an anonymous function can take two arguments, for example:
possible_moves = [
 	Coordinate( 1, 0), 
 	Coordinate( 0, 1), 
 	Coordinate(-1, 0), 
 	Coordinate( 0,-1),
]
function trajectory(c::Coordinate, n::Int, L::Number)
	return accumulate((c1,c2)->collide_boundary(c1+c2,L), vcat(c,rand(possible_moves, n)))
end

## %% Simple plot example of many random walks, using the trajectory function above
using Plots
function plot_trajectory!(p::Plots.Plot, trajectory::Vector; kwargs...)
	plot!(p, (c->(c.x,c.y)).(trajectory); 
		label=nothing, 
		linewidth=2, 
		linealpha=LinRange(1.0, 0.2, length(trajectory)),
		kwargs...)
end
let
	p = plot(ratio=1)
	for _ in 1:10
		long_trajectory = trajectory(Coordinate(), 1000, 20)
		plot_trajectory!(p, long_trajectory)
	end
	p
end

## %% Floor a number and convert to Int
floor(Int, L)

## %% Generated functions generate specialised code
# depending on the types of the arguments, with more
# flexibility and/or less code than multiple dispatch.
# Generated function gets expanded at a time when the
# types of the arguments are known, but the function is
# not yet compiled.
# In the body we only have access to types, not values.
# Returns a quoted expression, that is compiled and then run.
# They must not mutate or observe non-constant global states.
@generated function foo(x)
  Core.println(x)
  return :(x * x)
end
foo(2)
foo("bar")
foo(4)     # The result foo(2) was cached 
foo("bat") # The result foo("bar") was also cached

## %% You can replace the default constructor in a struct
struct MyThing
  x::Int64
  MyThing(x) = (x <= 0) ? error("no good") : new(x)
end
struct MyThing2
  x::Int64
  MyThing2() = new(1)
end
struct MyThing3
  x::Int64
end
MyThing3() = MyThing3(1)

## %% "Static type" and "dynamic tag" are both referred to
# as 'type' (confusingly)
function f()::Union{Int, String} # static type
  if ( rand() < 0.5 ) 
    return 1
  else 
    return "1"
  end
end

x = f() # x now has dynamic type either Int or String, not the union.
typeof(x)

## %% Broadcasting on multiple arguments
f(x::Int64,y::Int64,z::Int64) = x + y + z
g(x::Int64,y::Vector{Int64},z::Int64) = x + sum(y) + z
# f will broadcast over whatever arguments are passed as vectors
f.([1,2,3], 1, 1)
f.([1,2,3], [1,2,3], 1)
# even though parameter y of g is a vector type, g will broadcast over y
g.(1, [1,2], 1)

## %% Instead of sum(a .== b) we can use count(predicate, iter)
count(==(1), [1,2,1,3,1,1])

## %% Returning a function is useful for plotting derivatives
function finite_difference_slope(f::Function, a, h=1e-3)
	return (f(a+h)- f(a))/h
end
function tangent_line(f, a, h)
	m = finite_difference_slope(f,a,h)
	return x -> m*(x-a) + f(a)
end

## %% Generate a nxn random matrix
n = 5
M = rand(n,n)

## %% Upper and lower triangles
using LinearAlgebra
M = rand(5,5)
triu(M,1) # starting from the 1st superdiagonal (note 0 gives diagonal too)
tril(M,-1) # and negative numbers give subdiagonals

## %% We can copy the upper triangle to the (0) lower triangle using max
using LinearAlgebra
M = rand(5,5)
M = triu(M,1)
M = max.(M,M')

## %% The identity matrix I is defined in LinearAlgebra
using LinearAlgebra
I
ones(Int, 3, 3) - I

## %% SparseArrays defines a sparse zeros matrix
using SparseArrays
spzeros(Bool, 3, 3)

## %% findfirst gives the first index in the array for which the condition is satisfied
k = 5
c = cumsum([0.1, 0.2, 0.1, 0.1, 0.4, 0.1])
r = rand(k)
map( x -> findfirst(c .> x), r )

## %% Packages from pagerank "important"
# HTTP
# Dates
# JSON
# DataFrames
# Requires
# Distributed
# DataStructures
# Random
# Compat
# Reexport
# StatsBase
# SparseArrays
# LinearAlgebra
# Statistics
# Distributions
# StaticArrays
# Test

## %% Dict comprehension
keys = ["a","b","c"]
Dict(keys[i] => i-1 for i = 1:length(keys))

## %% Reference (not copy) to DataFrames vector using bang (!)
# https://dataframes.juliadata.org/stable/lib/indexing/#getindex-and-view-1
using DataFrames
df = DataFrame(A = 1:4, B = ["M", "F", "F", "M"])
df[:,2] # This makes copy
df[!,2] # This makes a reference
df[!,2] .= 1 # So, e.g., I can set with side effects