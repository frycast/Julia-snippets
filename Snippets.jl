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
dump(expr2)

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