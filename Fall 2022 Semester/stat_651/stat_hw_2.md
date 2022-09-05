1. (Casella/Berger Problem 1.1) For each of the following experiments, describe  
   the sample space. Also, state whether the sample space is finite, countably  
   infinite, or uncountably infinite.
   
   
   
   1. Toss a coin four times.
      
      A sample space is "the set $S$ of all possible outcomes of a particular experiement."  In this case, our experiment consists of tossing a coin four times.  That means for each toss we have an outcome $H$ or $T$ which stack together for the entire experiment we see something like:
      
      $\{HTHT\}$ 
      
      In this example, the first flip yielded heads, the second tails, the third heads, and the fourth tails
      
      in $\{TTTT\}$ 
      
      all four flips resulted in tails..
      
      The total number of singletons in the sample space is $2^{4} = 16$.
      
      Since there are a finite number of possibilities in the sample space, the sample space is finite.
      
      
   
   2. Count the number of insect-damaged leaves on a plant.
      
      For this one to make sense, we have to think about the experiment a little bit.  We take in a "plant" as an experiment, and we spit out a value $a$ where $a$ is the number of damaged leaves.
      
      
      
      This one is tricky, because "what is the maximum number of leaves a single plant can have?" is an important part of the equation here.  Or "what is a posible outcome of the experiment."  But if we ignore that first, we see that the sample space is a set of numbers of damaged leaves, like "$4$" or "$23$" or $2^{234} - 10$.
      
      then the sample space $\mathcal{S} = \{a \text{ such that }a\in \mathbb{Z}^+ \cup 0 \}$
      
      That is, "The set  of all positive integers $a$ or $0$" provided there is no single "leafiest" plant out there in the unverse.  That would make it countably inifinite.  That said, that doesn't feel right to me.
      
      
      
      There probably is a "leafist plant in existence," or at least a "plant with the most possible leaves" or if you know about all the plants you plan on looking at your experiment, you can say that the sample space *is* finite for your purposes.  So you could say
      
      $\mathcal{S} = \{a\text{ such that }0\leq a \leq m \text{ for some }a, m\in \mathbb{Z}^+ \cup 0 \}$
      
      That is, "the set of integers $a$ such that $0 \leq a \leq m$ for some non-negative integer $a$ and $m$, where $a$ is the number of damaged leaves and $m$ is the maximum number of leaves on the leafiest plant in your experiment."
      
      In this case, both $a$ and $m$ are known before we start looking at leaves, and there are a finite number of elements of $\mathcal{S}$ because there are a finite number of leaves one can even look at to begin with.  In this case the sample space would be finite.  This seems less problematic to me, because there is the possibility that the experiment records a plant where literally every leaf was damaged, but it isn't possible that the experiment records values in excess of the largest number of leaves a plant has.
      
      
   
   3. Measure the lifetime (in hours) of a particular brand of light bulb.
      
      Provided we aren't looking at decimal hours of arbitrary precision and are only concerned with lightbulbs that lasted an integer number of hours (that is if a light bulb lasted 2074.3 hours we count it as 2074 hours etc), I think the case could be made that this is countably infinite.  We could have a lightbulb that lasted an arbitrarily high amount of hours.  A lightbulb that burns out in 89 million hours is *possible* though unlikely.  The set of all possible outcomes contains all the positive integers and zero, hence $\mathcal{S}$ is countably infinite.
   
   4. Record the weights of 10-day-old rats.
      
      This again is related to the precision of our measuring tools, but provided that a 10-day-old rat's length can be a real number betwen $a, b \in \mathbb{R}^+ \cup 0$ such that $a \neq b$ I'd say this is "uncountably inifinite."  If we are forced to round to some nearest fraction because of our tools, then this becomes countably infinite (because the rational numbers and the integers have the same cardinality) but for all practical purposes, the length of a baby rat is a real number, so I'd say this is uncountably infinite.
   
   5. Observe the proportion of defectives in a shipment of electronic components.
      
      This seems similar to the leaf problem.  Each experiment is a shipment of electronic components, and the value returned from that experiment is rational number of $\frac{\text{defective components}}{\text{total components}}$.  If our experiment is for any hypothetical amount of components in any shipment in the uniiverse and the total number of components is some arbitrary integer greater than zero, then the sample space is countably infinite because you could get any rational number greater than zero.
      
      This seems wrong to me though, because - really if you're looking at a specific shipment of electronic components, then $\mathcal{S}$ is finite - it's bounded by the size of the shipment.  In a simple case, suppose that there are only 3 possible components in the shipment.  Then: 
      
      $\mathcal{S} = \{\frac{0}{3}, \frac{1}{3}, \frac{2}{3}, \frac{3}{3} \}$
      
      Which is decidedly countable - namely, $|\mathcal{S}| = 4$.  
      
      

2. Casella/Berger Problem 1.4.) For events A and B, find formulas for the prob-  
   abilities of the following events in terms of the quantities P (A), P (B) and  
   P (A ∩ B). (These take a bit of thinking; I find that it helps me sort them out  
   if I sketch a Venn diagram for them.)
   
   1. either $A$ or $B$ or both.
      
      $P(A) + P(B) - P(A \cap B)$
   
   2. either A or B but not both.
      
      $P(A) + P(B) - 2P(A \cap B)$
      
      
   
   3. ??????????at least one of $A$ or $B$
      
      The probability of at least one $A$ or $B$ is the same as the probability of $1 - P((A\cup B)^c )$
      
      $1 - (P(A) + P(B)) = 1 - P(A) - P(B)$
   
   4. at most one of $A$ or $B$

3. (Casella/Berger Problem 1.5.) Approximately 1/3 of all human twins are iden-  
   tical (one-egg) and 2/3 are fraternal (two-egg) twins. Identical twins are neces-  
   sarily the same sex, with male and female being equally likely. Among fraternal  
   twins, approximately 1/4 are both female, 1/4 are both male, and half are one  
   male and one female. Finally, among all U.S. births, approximately 1 in 90 is a  
   twin birth. Define the following events:
   
   
   A = {a U.S. birth results in twin females}  
   B = {a U.S. birth results in identical twins}  
   C = {a U.S. birth results in twins}  
   
   
   
   1. State, in words, the event A ∩ B ∩ C.  
      
      The set of identical twin female births in the US.
   
   2. Find P (A ∩ B ∩ C)
   
   $P(A\cap B \cap C) = \frac{1}{90} \frac{1}{3} \frac{1}{2} = \frac{1}{540}$

4. Casella/Berger Problem 1.2d) Verify the following identity (by showing that  
   each side is a subset of the other side):  
   A ∪ B = A ∪ (B ∩ $A^c$)  
   
   
   Suppose $A$ and $B$ are sets.
   
   
   
   Let $x \in A\cup B$.
   
   Then $x \in A$ or $x \in B$
   
   if $x\in A$ then $x \in A \cup (B \cap A^c)$ because $x \in A$.
   
   if $x \in B$ then $x \in B \cap A^c$ so $x \in A \cup (B \cap A^c)$
   
   either way, $x \in A \cup (B \cap A^c)$
   
   hence, $A \cup B \subseteq A \cup (B \cap A^c)$.
   
   
   
   What if, on the other hand, we let $x \in A \cup (B \cap A^c)$.
   
   Then $x \in A$ or $x \in (B \cap A^c)$.
   
   if $x\in A$, then $x \in A \cup B$.
   
   if $x \in (B \cap A^c)$, then $x \in B$ so $x \in A \cup B$.
   
   Either way, $x \in A \cup B$, so
   
   $A \cup (B \cap A^c) \subseteq A \cup B$.
   
   
   
   Therefore $A \cup (B \cap A^c) = A \cup B$.
   
   
   
   
   
   Sketch a Venn diagram to illustrate.
   
   

5. If P (A) = 1/3 and P ($B^c$) = 1/4, can events A and B be disjoint? (Casella/Berger  
   problem 1.13.)
   
   $P(A) = \frac{1}{3}$ and $P(B^c) = \frac{1}{4}$
   
   so $P(B) = \frac{3}{4}
   $
   
   Suppose $A \cap B = \emptyset$ ($A$ and $B$ are disjoint.)
   
   Then $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ by theorem 1.2.2b.
   
   further, by Theorem 1.2.1a $P(A\cap B) = 0$
   
   so,
   
   $P(A \cup B) = P(A) + P(B) = \frac{1}{3} + \frac{3}{4} \geq 1$.  This is a contradiction!  Thus $A$ and $B$ cannot be disjoint.
   
   

5. 
   1. Sketch a Venn diagram to illustrate a sample space containing events A  
      and B such that P (A∩B) = 0.12, P (A\B) = 0.28, and P ((A∪B)c) = 0.42.  
      
      ![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-05-13-27-44-image.png)
   2. Find (calculate) P (A), P (B), P (A ∪ B), and P (Ac ∩ B).
      
      $P(A) = P(A \setminus B) + P(A \cap B) = 0.28 + 0.12$
      
      $P(B) = 1 - P((A \cup B)^c) - P(A\setminus B) = 1 - 0.42 - 0.28= 0.30$
      
      $P(A \cup B) = 1 - P((A \cup B)^c) = 1 - 0.42 = 0.58$
      
      
6. Find (calculate) the number of elements there are in the sigma algebra B for  
   the following sample spaces S. (You need to justify your answers by describing  
   and/or listing some of the outcomes in the sample space.)  
   1. S = { outcomes obtained by flipping 4 distinct coins once } (e.g. penny,  
      nickel, dime, quarter)  
      
      
      
      You have 2 choices per coin, and you flip 4 different coins.  This is "ordered" - that is to say each experiment has an associated coin, and each one has 2 possible states.  That means that the sample space has $2^4 = 16$ entries, and the Sigma Algebra $|\mathcal{B}| = 2^{16}
      $
      
      
   2. S = { outcomes obtained by flipping 4 identical coins once } (e.g. four  
      1919 pennies)  
      
      The coins are identical, so they're unordered with replacement.
      
      so, $|\mathcal{S}| = \frac{5!}{2! 3!} = 10$ but we don't count $\{HHTTT\}$ any different from $\{TTHH\}$, so divide that by two again so really $|\mathcal{S}| = 5$.  That means the size of $\mathcal{B}$ is $2^5$.
      
      
      
      
   3. S = { outcomes obtained by flipping 4 coins once, where two of the coins  
      are (say) indistinguishable pennies, and the other two are (say) indistin-  
      guishable quarters }  
      
      So, this one is a little easier to draw out.
      
      ![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-05-13-58-14-image.png)
      
      So we have "coin type 1" and "coin type 2" in this experiment.  Coin type one can have one of the values that can be found in column one, and each one of those has the possibility of being followed by all four of the results in column two.
      
      Ultimately, that means that $|\mathcal{S}| = 16$ so $|\mathcal{B}| = 2^{16}$
   4. S = { outcomes obtained by flipping a single coin 4 times }
      
      It's a single coin, so we're looking at "ordered" with replacement, so $n^r = 4^2 = 16$ so $|\mathcal{B}| = 2^{16}$ 
7. 
   
   ![](/home/patrickpragman/.var/app/com.github.marktext.marktext/config/marktext/images/2022-09-05-14-05-39-image.png)
