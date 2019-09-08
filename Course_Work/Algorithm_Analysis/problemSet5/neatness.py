# Norman Hong

def neatest_page(M, W):
    penalty_subproblem = []
    lines = []    
    while len(W) != 0:
        temp = []
        
        for j in range(0, len(W)):
            penalty = M-j+0 - sum([len(k) for k in W[0:j+1]])
            
            if penalty > 0:
                temp.append(penalty)
        penalty_subproblem.append(min(temp))
        lines.append(W[0:len(temp)])
        del W[0:len(temp)]
        
    return lines, sum(penalty_subproblem[:-1])


def main(line_size):
    
    text = "Buffy the Vampire Slayer fans are sure to get their fix with the DVD release of the show's first season. The three-disc collection includes all 12 episodes as well as many extras. There is a collection of interviews by the show's creator Joss Whedon in which he explains his inspiration for the show as well as comments on the various cast members.  Much of the same material is covered in more depth with Whedon's commentary track for the show's first two episodes that make up the Buffy the Vampire Slayer pilot. The most interesting points of Whedon's commentary come from his explanation of the learning curve he encountered shifting from blockbuster films like Toy Story to a much lower-budget television series. The first disc also includes a short interview with David Boreanaz who plays the role of Angel. Other features include the script for the pilot episodes, a trailer, a large photo gallery of publicity shots and in-depth biographies of Whedon and several of the show's stars, including Sarah Michelle Gellar, Alyson Hannigan and Nicholas Brendon."
    text = text.split(" ")
    lines_buffy, penalty_buffy = neatest_page(line_size, text)
    
    
    for line in lines_buffy:
        print(' '.join(word for word in line))
    
    print('\n')
    print("Minimum Penalty = : ", penalty_buffy)	
    print('\n')

main(40)
main(72)

